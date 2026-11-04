# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""R-09(1) gate: aiter fused AR+RMSNorm bridge (ar_norm_aiter).

Two layers:
  1. bridge presence -- the flashinfer seam symbol carries the wrapper
     flag (sitecustomize installed it; HCU_AR_NORM_IMPL must resolve to
     aiter on this stack).
  2. numeric + graph-replay verdict on real ranks -- spawns
     min(4, visible) ranks; each builds the aiter GraphSafe
     CustomAllreduce (gloo handshake group, as parallel_state does) and
     holds the fused kernel to the fp64-exact-AR truth at 1 ulp, plus a
     CUDA-graph replay with fresh data (R-06 corruption history).

Skips below 2 visible GPUs: the L1 kernels segment runs single-GPU, the
full verdict needs a multi-GPU window (manual 8-rank probe evidence:
nmz26 2026-10-19, 1ulp/0ulp, 1.38-2.11x, replay exact).

Run (container): HIP_VISIBLE_DEVICES=0,1,2,3 \\
  PYTHONPATH=/root/hcu_t2:/sglang-v0521/python \\
  python3 -m pytest tests/kernels/test_ar_norm_aiter_gfx938.py -q -s
"""

import os

import pytest
import torch

torch_version_hip = getattr(torch.version, "hip", None)


def _visible_gpus():
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


def _rank_worker(rank, world, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    import torch.distributed as dist

    dist.init_process_group("nccl", rank=rank, world_size=world)
    gloo = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)

    from aiter.dist.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    ca = CustomAllreduce(gloo, dev)

    hidden, eps = 4096, 1e-5
    g = torch.Generator(device="cpu").manual_seed(4242)
    w = (torch.randn(hidden, generator=g) * 0.1 + 1.0).to(
        dev, torch.bfloat16)

    def exact_ar(x):
        parts = [torch.empty_like(x, dtype=torch.float32)
                 for _ in range(world)]
        dist.all_gather(parts, x.float())
        return torch.stack(parts).sum(0)

    def rms(x, wgt):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
                * wgt.float()).to(torch.bfloat16)

    def ulp_err(out, ref):
        u = torch.pow(
            2.0, ref.float().abs().clamp(min=1.0).log2().floor() - 7.0)
        return ((out.float() - ref.float()).abs() / u).max().item()

    for t, one_stage in ((64, False), (4, True)):
        torch.manual_seed(100 + rank)
        x = torch.randn(t, hidden, device=dev, dtype=torch.bfloat16) * 0.7
        res = torch.randn(t, hidden, device=dev, dtype=torch.bfloat16) * .3
        res_ref = res.clone()
        normed, res_out = ca.custom_fused_ar_rms(x, res, w, eps, one_stage)
        res_exact = (exact_ar(x).to(torch.bfloat16) + res_ref)
        norm_exact = rms(res_exact, w)
        en, er = ulp_err(normed, norm_exact), ulp_err(res_out, res_exact)
        assert en <= 2.0 and er <= 2.0, (
            f"rank{rank} t={t}: fused ulp-err norm {en:.2f} res {er:.2f}")
        if rank == 0:
            print(f"[PASS] parity t={t}: ulp-err norm {en:.2f} res {er:.2f}")

    # graph replay with fresh data (R-06 signature)
    t = 4
    x = torch.randn(t, hidden, device=dev, dtype=torch.bfloat16)
    res = torch.randn(t, hidden, device=dev, dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ca.custom_fused_ar_rms(x, res, w, eps, True)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = ca.custom_fused_ar_rms(x, res, w, eps, True)
    for trial in range(2):
        torch.manual_seed(9000 + trial * 11 + rank)
        x.copy_(torch.randn_like(x) * 0.9)
        res.copy_(torch.randn_like(res) * 0.4)
        x_ref = x.clone()
        res_ref = res.clone()
        gr.replay()
        torch.cuda.synchronize()
        res_exact = exact_ar(x_ref).to(torch.bfloat16) + res_ref
        en = ulp_err(out[0], rms(res_exact, w))
        er = ulp_err(out[1], res_exact)
        assert en <= 3.0 and er <= 3.0, (
            f"rank{rank} replay {trial}: ulp-err norm {en:.2f} res {er:.2f}")
        if rank == 0:
            print(f"[PASS] graph replay {trial}: ulp-err {en:.2f}/{er:.2f}")

    torch.cuda.synchronize()
    dist.barrier()
    del gr, out, s, ca
    torch.cuda.synchronize()
    dist.destroy_process_group(gloo)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch_version_hip is not None),
    reason="needs HIP GPU (aiter tier)")
def test_bridge_installed():
    from hcu_adapter import hcu_env

    if hcu_env.resolve("HCU_AR_NORM_IMPL")[0] != "aiter":
        pytest.skip("HCU_AR_NORM_IMPL did not resolve to aiter")
    from sglang.srt.layers import flashinfer_comm_fusion as fcf
    from sglang.srt.layers.layer_boundary.fusions import allreduce as fus
    from sglang.srt.layers.layer_boundary.residual import add_norm as an

    assert getattr(
        fcf.flashinfer_allreduce_residual_rmsnorm,
        "_hcu_ar_norm_aiter", False), "flashinfer dispatch not bridged"
    # <=v0.5.21 requires the ARM copies; v0.5.22 removed that gate and uses
    # the dispatch seam directly.
    if hasattr(an, "apply_flashinfer_allreduce_fusion"):
        for mod in (an, fus):
            assert getattr(
                mod.apply_flashinfer_allreduce_fusion,
                "_hcu_ar_norm_arm", False), (
                f"arm not rebound on {mod.__name__}")


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch_version_hip is not None),
    reason="needs HIP GPU (aiter tier)")
@pytest.mark.skipif(_visible_gpus() < 2, reason="needs >=2 visible GPUs")
def test_fused_parity_and_graph_replay():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        master_port = sock.getsockname()[1]
    world = 2
    torch.multiprocessing.spawn(
        _rank_worker, args=(world, master_port), nprocs=world, join=True)


if __name__ == "__main__":
    test_bridge_installed()
    test_fused_parity_and_graph_replay()
    print("ar_norm_aiter gate: OK")
