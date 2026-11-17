# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import socket
import os

import pytest
import torch
import torch.multiprocessing as mp


def _worker(rank, port):
    os.environ.update(LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE='2')
    from sglang.multimodal_gen.runtime.platforms import initialize_current_platform
    from sglang.multimodal_gen.runtime.distributed import parallel_state as native
    torch.cuda.set_device(rank)
    initialize_current_platform()
    assert native.get_local_torch_device() == torch.device('cuda', rank)
    native.init_distributed_environment(world_size=2, rank=rank, local_rank=rank, backend='nccl', distributed_init_method=f'tcp://127.0.0.1:{port}', device_id=torch.device('cuda', rank), timeout=120)
    native.initialize_model_parallel(tensor_parallel_degree=2, sequence_parallel_degree=1)
    group = native.get_tp_group()
    assert group.world_size == 2 and group.srt_custom_allreduce is None
    assert torch.distributed.get_backend(group.device_group) == 'nccl'
    for dtype in [torch.float32, torch.bfloat16]:
        for size in [4096, 1048576]:
            base = torch.arange(size, dtype=torch.int32).remainder(17)
            source = (base + rank).to(device='cuda', dtype=dtype)
            def execute():
                return group.all_reduce(source.clone())
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2): result = execute()
            stream.synchronize()
            torch.distributed.barrier(group=group.cpu_group)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream): captured = execute()
            for phase in range(3):
                source.copy_((base + 3 * phase + rank).to(dtype))
                expected = (2 * base + 6 * phase + 1).to(dtype)
                result = execute()
                torch.cuda.synchronize()
                assert torch.equal(result.cpu(), expected)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(captured.cpu(), expected)
            del graph
    print(f'PASS native TP2 RCCL eager and changed-input graph rank={rank}', flush=True)
    native.destroy_model_parallel()
    native.destroy_distributed_environment()


def test_native_diffusion_tp2_rccl_off_policy_eager_and_graph(monkeypatch):
    if torch.version.hip is None or torch.cuda.device_count() < 2:
        pytest.skip('requires two HCU devices')
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx936'):
        pytest.skip('current diffusion candidate is gfx936')
    monkeypatch.setenv('HCU_AITER_AR', '0')
    monkeypatch.setenv('SGLANG_DIFFUSION_PLATFORM_OVERRIDE', 'hcu')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)
