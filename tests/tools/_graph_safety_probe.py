# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-graph safety probe for HCU bridge kernels (decode-shaped).

For each bridge fn: capture a graph around it, then overwrite the INPUT
buffer contents (new positions / scores / lengths) and replay. If the
output tracks the NEW inputs -> graph-safe (no baked host state). If it
stays at capture-time values -> an H2D/host-side bake broke replay —
the classic '2nd decode token garbage' signature.
"""
import os
import sys

sys.path.insert(0, os.environ.get("HCU_ADAPTER_ROOT", "/adapter"))
import torch

dev = "cuda"
torch.manual_seed(0)
print(f"torch {torch.__version__}")

failures = []


def run_graph_probe(name, fn, make_inputs, mutate, out_of):
    """make_inputs() -> list of GPU tensors (fixed buffers, deterministic);
    mutate(bufs, step) writes step-specific data into them;
    out_of(result) -> comparable tensor."""
    torch.manual_seed(1234)
    bufs = make_inputs()

    # eager reference for step A (capture data) and step B (mutated data)
    eager_a = out_of(fn(*[b.clone() for b in bufs]))
    mutate(bufs)
    eager_b = out_of(fn(*[b.clone() for b in bufs]))

    # fresh deterministic buffers identical to step A for capture
    torch.manual_seed(1234)
    bufs = make_inputs()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*[b.clone() for b in bufs])  # clones: in-place safe warmup
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        result = fn(*bufs)

    # replay with capture-time data (sanity), then mutated data
    g.replay()
    torch.cuda.synchronize()
    got_a = out_of(result).clone()

    mutate(bufs)  # same mutation as eager_b
    g.replay()
    torch.cuda.synchronize()
    got_b = out_of(result).clone()

    def close(x, y):
        return (x - y).abs().max().item() <= 2e-2 * max(
            1.0, y.abs().max().item())

    ok_a = close(got_a.float(), eager_a.float())
    ok_b = close(got_b.float(), eager_b.float())
    status = "OK" if (ok_a and ok_b) else ("BAKED(replay!=new-input)"
                                           if ok_a and not ok_b else "WRONG")
    print(f"{name}: replay(capture-data)={ok_a} replay(new-data)={ok_b}"
          f"  -> {status}")
    if status != "OK":
        failures.append(name)
        print(f"   eager_new[:6] {eager_b.float().flatten()[:6].tolist()}")
        print(f"   got_new [:6] {got_b.float().flatten()[:6].tolist()}")


# ---- 1. rope (GPT-J interleave torch path, what GLM-5 uses) ----
from sglang.srt.layers.rotary_embedding import get_rope

RD, NH, T = 64, 8, 1  # decode shape: single token, 8 heads
rope = get_rope(RD, rotary_dim=RD, max_position=64, base=1e6,
                is_neox_style=False, dtype=torch.bfloat16).to(dev)


def rope_call(q, k, pos):
    return rope(pos, q, k)


def rope_inputs():
    return [torch.randn(T, NH, RD, dtype=torch.bfloat16, device=dev),
            torch.randn(T, 1, RD, dtype=torch.bfloat16, device=dev),
            torch.zeros(T, dtype=torch.int64, device=dev)]


def rope_mutate(bufs, step=1):
    with torch.no_grad():
        bufs[0].copy_(torch.randn(T, NH, RD, dtype=torch.bfloat16,
                                  device=dev))
        bufs[1].copy_(torch.randn(T, 1, RD, dtype=torch.bfloat16,
                                  device=dev))
        bufs[2].fill_(7)  # new decode position


run_graph_probe("rope_interleave", rope_call, rope_inputs, rope_mutate,
                lambda r: r[0].float())

# ---- 2. rmsnorm bridge (vllm._custom_ops.fused_add_rms_norm form) ----
import sgl_kernel  # loads adapter + sitecustomize bridges
import vllm._custom_ops as vco

H = 6144


def fused_call(x, res):
    out = torch.empty_like(x)
    rout = torch.empty_like(x)
    vco.fused_add_rms_norm(out, x, rout, res, _w, 1e-6)
    return out, rout


_w = (torch.randn(H, dtype=torch.bfloat16, device=dev) * 0.1 + 1.0)


def fused_inputs():
    return [torch.randn(2, H, dtype=torch.bfloat16, device=dev),
            torch.randn(2, H, dtype=torch.bfloat16, device=dev)]


def fused_mutate(bufs):
    with torch.no_grad():
        bufs[0].copy_(torch.randn(2, H, dtype=torch.bfloat16, device=dev))
        bufs[1].copy_(torch.randn(2, H, dtype=torch.bfloat16, device=dev))


run_graph_probe("fused_add_rms_norm_bridge", fused_call, fused_inputs,
                fused_mutate, lambda r: r[0].float())

# ---- 3. dsa topk bridge (fast_topk_v2 decode-shaped) ----
from sgl_kernel import fast_topk_v2

TT, W, KTOP = 2, 2048, 2048


def topk_call(score, lens):
    return fast_topk_v2(score, lens, KTOP, None)


def topk_inputs():
    return [torch.zeros(TT, W, dtype=torch.float32, device=dev),
            torch.tensor([1, 5], dtype=torch.int32, device=dev)]


def topk_mutate(bufs):
    with torch.no_grad():
        sc = torch.zeros(TT, W, dtype=torch.float32, device=dev)
        sc[0, 0] = 3.0  # row0 picks col0
        sc[1, 2] = 9.0
        sc[1, 4] = 8.0  # row1 picks col2, col4
        bufs[0].copy_(sc)


# topk with all-zero scores: picks first k columns
run_graph_probe("fast_topk_v2", topk_call, topk_inputs, topk_mutate,
                lambda r: r.float())

print("FAILURES:", failures if failures else "none — bridges are graph-safe")
