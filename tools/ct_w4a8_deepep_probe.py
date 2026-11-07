"""Two-rank synthetic CT INT8 expert/DeepEP communication regression.

Run in a fresh SGLang host with torch.distributed.run --nproc-per-node=2.
This exercises real dispatch/combine and the installed legacy expert core.
An optional mode also constructs the community dispatcher and its state machine.
Neither mode establishes checkpoint loading or model accuracy acceptance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS


def _source_binding(tree, module, relative):
    source = Path(module.__file__).resolve()
    if source != tree / relative:
        raise RuntimeError("Imported community source does not match selected tree")
    raw = source.read_bytes()
    git = ["git", "-c", f"safe.directory={tree}", "-C", str(tree)]
    if raw != subprocess.check_output(git + ["show", f"HEAD:{relative}"]):
        raise RuntimeError("Community source differs from Git HEAD")
    return source, raw


def run(args):
    import torch
    import torch.distributed as dist

    from hcu_adapter import hcu_env

    if (hcu_env.resolve("HCU_ADAPTER_HOST")[0] != "sglang"
            or hcu_env.resolve("HCU_W4A8_MOE")[0] != "int8"):
        raise RuntimeError("Requires fresh SGLang host with explicit INT8 candidate")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("Launch with torch.distributed.run --nproc-per-node=2")
    if args.recv_hook and not args.community_dispatcher:
        raise ValueError("--recv-hook requires --community-dispatcher")
    from deep_ep import Buffer
    from sglang.srt.layers.moe.ep_moe import layer as ep
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import deepep

    from hcu_adapter.kernels.w4a8_int8_native_triton import (
        w4a8_int8_native_masked_experts,
    )
    from tools.ct_w4a8_reference import int32_native_expert

    tree = args.up_tree.resolve()
    bindings = [
        _source_binding(tree, module, "python/sglang/srt/layers/moe/" + relative)
        for module, relative in (
            (ep, "ep_moe/layer.py"), (deepep, "token_dispatcher/deepep.py"),
        )
    ]
    commit = subprocess.check_output([
        "git", "-c", f"safe.directory={tree}", "-C", str(tree), "rev-parse", "HEAD",
    ], text=True).strip()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    buffer = None
    try:
        experts, hidden, intermediate, tokens, capacity = 4, 5120, 16, 4, 8
        rng = torch.Generator().manual_seed(115)

        def weight(outputs, columns):
            unsigned = torch.randint(0, 16, (experts, outputs, columns), generator=rng)
            packed = (unsigned[..., ::2] | unsigned[..., 1::2] << 4).byte()
            words = sum(unsigned[..., i::8].long() << (4 * i) for i in range(8)).int()
            scales = (torch.rand(experts, outputs, 1, generator=rng) * .003 + .001).bfloat16()
            return packed, scales, words

        gate, gs, gate_words = weight(2 * intermediate, hidden)
        down, ds, down_words = weight(hidden, intermediate)
        local = slice(rank * 2, rank * 2 + 2)
        layer = NS(
            _hcu_ct_channel_token_int8=True,
            w13_weight_packed=gate[local].cuda(),
            w2_weight_packed=down[local].cuda(),
            w13_weight_scale=gs[local].cuda(),
            w2_weight_scale=ds[local].cuda(),
            moe_runner_config=MoeRunnerConfig(
                is_gated=True, activation="silu", routed_scaling_factor=1.5,
                inplace=True, swiglu_limit=args.swiglu_limit,
            ),
        )
        value = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        ids = torch.empty(tokens, 2, device="cuda", dtype=torch.int64)
        weights = torch.empty(tokens, 2, device="cuda", dtype=torch.float32)
        dispatcher = None
        if args.community_dispatcher:
            from sglang.srt.environ import envs
            from sglang.srt.layers.moe.topk import StandardTopKOutput
            from sglang.srt.layers.moe.utils import DeepEPMode
            from sglang.srt.runtime_context import SpawnRanks, publish
            from sglang.srt.server_args import ServerArgs

            # Publish real configuration through the official API. "dummy"
            # avoids checkpoint I/O; this probe only uses synthetic weights.
            publish(ServerArgs(
                model_path="dummy", tp_size=2, ep_size=2,
                moe_a2a_backend="deepep", deepep_mode="low_latency",
                deepep_dispatcher_output_dtype="bf16",
            ), role="scheduler", ranks=SpawnRanks(rank, gpu_id=rank))
            if envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get() != capacity:
                raise ValueError(
                    "Community probe requires "
                    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8"
                )
            dispatcher = deepep.DeepEPDispatcher(
                dist.group.WORLD, router_topk=2, num_experts=experts,
                num_local_experts=2, hidden_size=hidden, params_dtype=torch.bfloat16,
                deepep_mode=DeepEPMode.LOW_LATENCY, return_recv_hook=args.recv_hook,
            )
        else:
            size = Buffer.get_low_latency_rdma_size_hint(capacity, hidden, 2, experts)
            buffer = Buffer(dist.group.WORLD, 0, size, low_latency_mode=True,
                            num_qps_per_rank=2, explicitly_destroy=True)
        # Compile the exact compute shape before any rank enters communication.
        w4a8_int8_native_masked_experts(
            torch.zeros(2, capacity * 2, hidden, device="cuda", dtype=torch.bfloat16),
            layer.w13_weight_packed, layer.w2_weight_packed,
            layer.w13_weight_scale, layer.w2_weight_scale,
            torch.zeros(2, device="cuda", dtype=torch.int32),
            swiglu_limit=args.swiglu_limit,
        )
        torch.cuda.synchronize()
        dist.barrier()

        def change(phase):
            generator = torch.Generator().manual_seed(116 + rank + phase * 10)
            value.copy_(torch.randn(tokens, hidden, generator=generator).bfloat16().cuda())
            routes = (torch.arange(tokens * 2).reshape(tokens, 2) + rank + phase) % experts
            if phase == 2:
                routes.fill_(-1)
            ids.copy_(routes.cuda())
            weights.copy_(torch.tensor(
                [[.25, .75] if phase == 0 else [.5, .25]] * tokens, device="cuda",
            ))

        def compute():
            nonlocal buffer
            if dispatcher is not None:
                dispatch = dispatcher.dispatch(
                    value, StandardTopKOutput(weights, ids, None),
                )
                buffer = deepep.DeepEPBuffer._state().buffer
                if (dispatch.hidden_states.dtype != torch.bfloat16
                        or dispatch.hidden_states_scale is not None):
                    raise RuntimeError("Community dispatcher did not preserve BF16 input")
                return dispatcher.combine(ep.DeepEPMoE.run_moe_core(layer, dispatch))
            received, counts, handle, event, _ = buffer.low_latency_dispatch(
                value, ids, capacity, experts, topk_weights=weights,
                use_fp8=False, async_finish=True,
            )
            event.current_stream_wait()
            dispatch = deepep.DeepEPLLDispatchOutput(
                received, None, ids, weights, counts,
                (tokens * 2 * 2 + experts) // experts,
            )
            combine = ep.DeepEPMoE.run_moe_core(layer, dispatch)
            result, event, _ = buffer.low_latency_combine(
                combine.hidden_states, combine.topk_ids, combine.topk_weights,
                handle, async_finish=True,
            )
            event.current_stream_wait()
            return result

        checks = []

        def check(result, label):
            cpu, routes, ws = value.cpu(), ids.cpu(), weights.cpu()
            expected = torch.zeros(tokens, hidden)
            for expert in range(experts):
                positions = (routes == expert).nonzero()
                if len(positions):
                    rows, slots = positions.unbind(1)
                    output = int32_native_expert(
                        cpu[rows].unsqueeze(0),
                        (gate_words[expert, :intermediate], gs[expert, :intermediate]),
                        (gate_words[expert, intermediate:], gs[expert, intermediate:]),
                        (down_words[expert], ds[expert]),
                        swiglu_limit=args.swiglu_limit,
                    )[0]
                    expected.index_add_(0, rows, output.float() * ws[rows, slots, None])
            expected, actual = expected.bfloat16(), result.cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            checks.append(dict(phase=label, max_abs=float(
                (actual.float() - expected.float()).abs().max(),
            )))

        change(0)
        check(compute(), "eager")
        dist.barrier()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            compute()
            compute()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = compute()
        for phase in range(4):
            change(phase)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            check(result, f"graph-{phase}")
        for source, raw in bindings:
            if source.read_bytes() != raw:
                raise RuntimeError("Community source changed during communication probe")
        args.out.mkdir(parents=True, exist_ok=True)
        report = dict(
            rank=rank, checks=checks, communication_passed=True,
            model_acceptance=False, swiglu_limit=args.swiglu_limit,
            community_dispatcher=args.community_dispatcher, recv_hook=args.recv_hook,
            community_commit=commit,
            sources={str(path.relative_to(tree)): hashlib.sha256(raw).hexdigest()
                     for path, raw in bindings},
            scope="Synthetic weights; real BF16 dispatch, INT8 expert core and combine",
        )
        # Refuse to overwrite evidence from a preceding run.
        with (args.out / f"rank{rank}.json").open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
        dist.barrier()
    finally:
        if buffer is not None:
            torch.cuda.synchronize()
            if args.community_dispatcher:
                # The community facade owns a buffer with automatic destruction.
                deepep.DeepEPBuffer._state().buffer = None
                buffer = None
            else:
                buffer.destroy()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--up-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="External directory for per-rank evidence")
    parser.add_argument("--swiglu-limit", type=float)
    parser.add_argument("--community-dispatcher", action="store_true",
                        help="Construct the real community dispatcher and runtime config")
    parser.add_argument("--recv-hook", action="store_true",
                        help="Exercise the community dispatcher's receive-hook mode")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
