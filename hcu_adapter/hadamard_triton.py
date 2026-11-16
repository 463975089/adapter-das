# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Triton butterfly Hadamard transform for the DSA indexer rotate_activation.

Upstream CUDA JIT kernel (sglang.kernels.ops.quantization.hadamard) compiles
warp-shuffle intrinsics the DTK toolchain rejects, and the HIP arm of the
caller imports ``fast_hadamard_transform``, which this wheel set lacks. The
torch emu (hadamard_emu.py) multiplies by a materialized Sylvester matrix
(n x n fp32 — 64MB at n=4096, built once but streamed every call); the
transform itself is a radix-2 butterfly, so a Triton kernel does it with
O(n log n) traffic and no matrix.

Per radix round with pair stride ``bit`` the flat row is viewed as
``[q, e, r]`` (e = the pairing bit); split e into lo/hi, emit lo+hi / lo-hi,
join back — validated against the Sylvester product exactly at n=2/4 and to
float32 addition-order noise (<=1e-6 rel) at n>=8 (numpy oracle, 2026-11-05).
Rounds use tl.reshape/permute/split/join only — no cross-lane shuffle
opcodes, which is exactly what the CUDA warp-shuffle JIT needed and DTK
rejects.

Caller contract (dsa_indexer.rotate_activation): power-of-2 last dim
(upstream asserts), scale=n**-0.5, fp32 compute regardless of io dtype
(same rounding posture as the emu), graph-capture safe (fixed shapes, no
host sync). The 12n/20n/28n/40n aliases route here too: the DSA chain only
ever calls the plain 2^k entry; the aliases keep the module-level rebinds
whole.

HCU_HADAMARD_IMPL=triton (default) | emu (torch reference, A/B probe).
"""

import logging

import torch
import triton
import triton.language as tl

LOG = logging.getLogger("hcu.adapter")

_MAX_N = 1 << 14


@triton.jit
def _butterfly_round(
    X,  # rows of N elems, contiguous, fp32; one radix-2 round in place
    n_rows,
    N: tl.constexpr,
    QB: tl.constexpr,  # N // (2*BIT), computed host-side (constexpr math
    BIT: tl.constexpr,  # in-body produced tensors on the DTK frontend)
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    offs = tl.arange(0, N)
    base = X + row.to(tl.int64) * N
    x = tl.load(base + offs).to(tl.float32)
    # flat row as [q, e, r] with e the pairing bit; split e into lo/hi,
    # emit lo+hi / lo-hi, join, restore the flat layout.
    lo, hi = tl.split(tl.permute(tl.reshape(x, (QB, 2, BIT)), (0, 2, 1)))
    x = tl.reshape(tl.permute(tl.join(lo + hi, lo - hi), (0, 2, 1)), (N,))
    tl.store(base + offs, x.to(X.dtype.element_ty))


def _triton_hadamard_transform(x: torch.Tensor, scale: float = 1.0, **_ignored):
    n = x.size(-1)
    if n & (n - 1) or n <= 0:
        raise ValueError(
            f"hadamard_triton: last dim must be a power of 2, got {n}"
        )
    if n > _MAX_N:
        raise ValueError(f"hadamard_triton: n={n} exceeds the {_MAX_N} budget")
    shape = x.shape
    # out-of-place for the caller; fp32 staging matches the emu rounding
    # (.to() is a no-op pass-through when x is already fp32+contiguous, so
    # the data_ptr check keeps the caller's tensor unmutated)
    buf = x.reshape(-1, n).to(torch.float32).contiguous()
    if buf.data_ptr() == x.data_ptr():
        buf = buf.clone()
    rows = buf.shape[0]
    warps = min(16, max(1, n // 512))
    bit = 1
    while bit < n:  # one kernel per radix round; log2(n) launches total
        _butterfly_round[(rows,)](
            buf, rows, N=n, QB=n // (2 * bit), BIT=bit, num_warps=warps
        )
        bit *= 2
    if scale != 1.0:
        buf *= scale
    return buf.to(x.dtype).reshape(shape)


def install():
    """Rebind the upstream hadamard ops module to the Triton butterfly."""
    from hcu_adapter import hcu_env, registry

    impl, src = hcu_env.resolve("HCU_HADAMARD_IMPL")
    if impl == "emu":
        from hcu_adapter.hadamard_emu import hadamard_transform as fn
    else:
        fn = _triton_hadamard_transform

    # 替代档: HCU_HADAMARD_IMPL 档挂载面（hadamard_transform 族只能在模块体上重绑）
    import sglang.kernels.ops.quantization.hadamard as had

    had.hadamard_transform = fn
    for nm in (
        "hadamard_transform_12n", "hadamard_transform_20n",
        "hadamard_transform_28n", "hadamard_transform_40n",
    ):
        setattr(had, nm, fn)
    registry.register(
        "hadamard",
        impl,
        target="seam:sglang.kernels.ops.quantization.hadamard.hadamard_transform",
        mechanism=registry.MECH_SEAM,
        detail="2^k butterfly, fp32 compute, caller tensor untouched"
        f" (resolved from {src})",
    )
    LOG.info("HCU adapter: hadamard_transform -> %s (%s)", impl, src)
