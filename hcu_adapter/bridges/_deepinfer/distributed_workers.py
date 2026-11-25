# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Externally launched CP workers for DeepInfer 774ce244."""

from hashlib import sha256
from pathlib import Path
import os
import sys

from hcu_adapter import registry

_SOURCE_PINS = {
    "engine/llm_engine.py": "a636d60ca5711f6b7c21e68370c3bd9c2de6fc3f4439a7429ca4f572f0edeebe",
    "entrypoints/llm.py": "069a5333d9f47047ccc87584792627dfa6eee9cf93bdc046107cbd0218a6bada",
    "model_executor/parallel_utils/parallel_state.py": "3565b2449abff4df0df2f979061a73364886c93602218b43c2f86c70f2629801",
    "model_executor/models/v4_modules/engram.py": "d2aa096a1f73177e2dd06f5a1fcd6746139c52af56c3bcfb77d5226a3deeae8f",
    "worker/base.py": "40e6ca29aebd9d475eaad4b86326783f76dba2b73feaa51d86a200b7704a05ed",
    "worker/worker.py": "ad32e595d9ab289dff1a7709136431ac60dae709e9e5123ae42eb0ebf7c30cbb",
}
_installed = False


class CommandQueue:
    buffer = None

    def __init__(self, group, writer_rank):
        from torch import distributed as dist

        self.group = group
        self.writer = dist.get_global_rank(group, writer_rank)
        self._is_writer = dist.get_rank(group) == writer_rank

    @classmethod
    def create_from_process_group(cls, pg, max_chunk_bytes, max_chunks, writer_rank=0):
        return cls(pg, writer_rank)

    def enqueue(self, command):
        from torch import distributed as dist

        dist.broadcast_object_list([command], src=self.writer, group=self.group)

    def dequeue(self):
        from torch import distributed as dist

        command = [None]
        dist.broadcast_object_list(command, src=self.writer, group=self.group)
        return command[0]


def install():
    global _installed
    if _installed:
        return
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    if (world, local_world) != (8, 4) or not 0 <= rank < world or local_rank != rank % local_world:
        raise ValueError("DeepInfer external workers require two ordered nodes of four ranks")
    if not os.environ.get("MASTER_ADDR") or not 50 < int(os.environ["MASTER_PORT"]) <= 65535:
        raise ValueError("DeepInfer requires a master address and a port in [51, 65535]")
    if not registry.get("deepinfer._C"):
        raise RuntimeError("Install the DeepInfer HCU runtime before installing external workers")
    upstream = Path(sys.modules["deepinfer"].__file__).resolve().parent
    for relative, digest in _SOURCE_PINS.items():
        if sha256((upstream / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Unsupported DeepInfer source for external workers: {relative}")

    from deepinfer.engine.llm_engine import LLMEngine
    from deepinfer.entrypoints import llm
    from deepinfer.model_executor.parallel_utils import parallel_state
    from deepinfer.utils import get_distributed_init_method
    from deepinfer.worker.runtime_config import expose_all_gpus_to_cuda_driver
    from deepinfer.worker.worker import EngineCallbacks, Worker
    from hcu_adapter.kernels.deepep_dtk import EngramBuffer

    # DeepInfer opens its own TCP store on MASTER_PORT - 50.
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)

    class DistributedEngine(LLMEngine):
        @classmethod
        def from_engine_args(cls, engine_args):
            if (engine_args.context_parallel_size, engine_args.expert_parallel_size,
                engine_args.engram_parallel_size, engine_args.data_parallel_size,
                engine_args.engine_rank) != (8, 8, 8, 1, 0):
                raise ValueError("External workers require one CP8/EP8/Engram8 engine")
            if rank == 0:
                return super().from_engine_args(engine_args)
            model, cache, parallel, scheduler = engine_args.create_engine_configs()
            expose_all_gpus_to_cuda_driver()
            Worker(engine_args.engine_type, model, parallel, scheduler, cache, local_rank,
                   rank, get_distributed_init_method(parallel), False)
            raise RuntimeError("Non-driver DeepInfer worker returned from its command loop")

        def _init_workers(self):
            callbacks = EngineCallbacks(post_process_fn=self.post_process,
                                        pre_sampler_sync_fn=self.pre_sampler_sync)
            self.driver_worker = Worker(
                self.engine_type, self.model_config, self.parallel_config,
                self.scheduler_config, self.cache_config, local_rank, rank,
                get_distributed_init_method(self.parallel_config), True, engine_callbacks=callbacks,
            )

    registry.replace(parallel_state, "MessageQueue", CommandQueue, namespace=__name__,
                     expected=parallel_state.MessageQueue)
    registry.replace(llm, "LLMEngine", DistributedEngine, namespace=__name__, expected=LLMEngine)
    # The pinned GPU Engram loader releases its own references after write().
    EngramBuffer.retain_gpu_storage = True
    _installed = True
