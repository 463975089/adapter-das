# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""R-05 bench: the deepgemm w4a16 MoE bridge vs the stock moe_wna16
triton arm, same tensors, eager, single card.

  HIP_VISIBLE_DEVICES=4 PYTHONPATH=/root/hcu_t2:/sglang-v0521/python \
  TVM_FFI_DISABLE_TORCH_C_DLPACK=1 python tests/tools/_bench_w4a16_moe.py
"""

import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig

# bare-script bench: publish nothing; stub the one read-only flag the
# triton-config lookup fetches from the runtime context
import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config as _fmc
import sglang.srt.runtime_context as _rc


class _D:
    enable_deterministic_inference = False


class _E:
    deterministic = _D()


class _C:
    enable_symm_mem = False


class _Any(_D):
    def __getattr__(self, item):
        return _Any()


class _EX:
    deterministic = _D()
    comm = _C()

    def __getattr__(self, item):
        return _Any()


_orig_bag = _rc.RuntimeContext.config_bag


def _bag(self, name, *a, **k):
    if name == "exec":
        return _EX()
    return _orig_bag(self, name, *a, **k)


_rc.RuntimeContext.config_bag = _bag
_fmc._CONTEXT_get_exec = None  # noqa: F841 (irrelevant, keep naming quiet)
try:
    _fmc.get_exec = lambda *_a, **_k: _EX()
except Exception:
    pass

# single-process bench has no TP context; publish a shim (same escape the
# R-10 probes used)
from sglang.srt.runtime_context import get_parallel


class _ShimGroup:
    rank = 0
    world_size = 1

    @property
    def device(self):
        return torch.device("cuda")


get_parallel().override_permanently(tp_group=_ShimGroup())

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    fused_experts,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput

from hcu_adapter.w4a16_moe import deepgemm_w4a16_fused_experts

dev = "cuda"


def make(E, I, K, M, topk, group, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    q1 = torch.randint(0, 16, (E, 2 * I, K), device=dev, generator=g)
    q2 = torch.randint(0, 16, (E, K, I), device=dev, generator=g)
    qw1 = (q1[:, :, 0::2] | (q1[:, :, 1::2] << 4)).to(torch.uint8)
    qw2 = (q2[:, :, 0::2] | (q2[:, :, 1::2] << 4)).to(torch.uint8)
    s1 = (torch.rand(E, 2 * I, K // group, device=dev, generator=g)
          * 0.02 + 0.005).to(torch.bfloat16)
    s2 = (torch.rand(E, K, I // group, device=dev, generator=g)
          * 0.02 + 0.005).to(torch.bfloat16)
    x = (torch.randn(M, K, device=dev, generator=g) * 0.3).to(torch.bfloat16)
    tids = torch.randint(0, E, (M, topk), device=dev, generator=g)
    tw = torch.softmax(torch.randn(M, topk, device=dev, generator=g), -1)
    return qw1, qw2, s1, s2, x, tw, tids


def timeit(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms


def main():
    torch.manual_seed(0)
    group = 128  # CT-int4 checkpoint downsizes to g32 halves; use wg grid
    # E, I, K, M, topk — chunked-prefill and decode-ish faces
    shapes = [
        (64, 1024, 4096, 8192, 8),
        (128, 1024, 4096, 8192, 8),
        (256, 512, 4096, 8192, 8),
        (64, 1024, 4096, 512, 8),
    ]
    print(f"{'E':>4} {'I':>5} {'K':>5} {'M':>5} | "
          f"{'triton ms':>10} {'dg ms':>10} {'x':>5} | maxabs parity")
    for E, I, K, M, topk in shapes:
        group = 128 if I % 128 == 0 else 32
        qw1, qw2, s1, s2, x, tw, tids = make(E, I, K, M, topk, group, E + M)
        cfg = MoeRunnerConfig(num_experts=E, activation="silu", is_gated=True,
                              inplace=False)
        tk = StandardTopKOutput(tw, tids, None)
        t_tri = timeit(lambda: fused_experts(
            x, qw1, qw2, tk, cfg, use_int4_w4a16=True, w1_scale=s1,
            w2_scale=s2,
            w1_zp=None, w2_zp=None,
            block_shape=[0, group]))
        out_dg = deepgemm_w4a16_fused_experts(x, qw1, qw2, tw, tids, s1, s2)
        t_dg = timeit(lambda: deepgemm_w4a16_fused_experts(
            x, qw1, qw2, tw, tids, s1, s2))
        out_tri = fused_experts(x, qw1, qw2, tk, cfg, use_int4_w4a16=True,
                                w1_scale=s1, w2_scale=s2, w1_zp=None,
                                w2_zp=None, block_shape=[0, group])
        mad = float((out_dg.float() - out_tri.float()).abs().max())
        print(f"{E:>4} {I:>5} {K:>5} {M:>5} | {t_tri:>10.2f} {t_dg:>10.2f} "
              f"{t_tri/t_dg:>5.2f} | {mad:.4f}")


if __name__ == "__main__":
    main()
