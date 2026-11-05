# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Hadamard butterfly (triton) vs Sylvester oracle — parity gates.

2026-11-05: HCU_HADAMARD_IMPL default emu→triton. The DSA indexer's
rotate_activation previously multiplied by a materialized n*n Sylvester
matrix (64MB at n=4096); the triton route is a radix-2 butterfly with no
warp-shuffle opcodes (the exact class of intrinsics that kept the upstream
CUDA JIT uncompilable on DTK).

Structure per testsuite convention:
  1. numpy Sylvester oracle — the step reference (CPU, always runs);
  2. parity gates on GPU: fp32 exact-tolerance, bf16 io with fp32 compute,
     3-D shapes, caller scale, and the emu arm (fallback tier);
  3. dual metrics cos + maxdiff; contract asserts (non-power-of-2 raises).

Run: python tests/kernels/test_hadamard_triton.py   (GPU; oracle-only on CPU)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def _impl():
    """Lazy: triton is GPU-host only; CPU runs cover the oracle alone."""
    try:
        from hcu_adapter import hadamard_triton

        return hadamard_triton
    except ImportError:
        return None


def _emu():
    from hcu_adapter.hadamard_emu import hadamard_transform as fn

    return fn


def _sylvester(n: int) -> np.ndarray:
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < n:
        H = np.concatenate(
            [np.concatenate([H, H], 1), np.concatenate([H, -H], 1)], 0
        )
    return H


def test_oracle_matches_sylvester_contract():
    # the reference itself: H(2k) orthonormal, H@H.T = n*I
    for n in (2, 8, 128):
        H = _sylvester(n)
        assert np.allclose(H @ H.T, n * np.eye(n), atol=1e-9), n


def test_non_power_of_two_is_refused():
    impl = _impl()
    if impl is None:
        print("SKIP test_non_power_of_two_is_refused (no triton)")
        return
    x = torch.zeros(3, 6)
    try:
        impl._triton_hadamard_transform(x)
    except ValueError as e:
        assert "power of 2" in str(e)
    else:
        raise AssertionError("non-power-of-2 last dim must raise")


def _parity(n, shape, dtype, scale, fn):
    rng = np.random.default_rng(n + shape[0])
    x_np = rng.standard_normal(shape).astype(np.float32)
    x = torch.from_numpy(x_np).to(dtype).cuda()
    ref = (
        (x_np.reshape(-1, n).astype(np.float64) @ _sylvester(n)) * scale
    ).reshape(shape)
    got = fn(x, scale=scale).float().cpu().numpy().astype(np.float64)
    cos = float(
        np.sum(got * ref)
        / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30)
    )
    rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
    return cos, rel


def _gpu_gate():
    if not (torch.cuda.is_available() and torch.version.hip):
        print("SKIP: no HIP GPU — oracle-only run")
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and name not in (
                "test_oracle_matches_sylvester_contract",
                "test_non_power_of_two_is_refused",
            ) and callable(fn):
                print(f"SKIP {name}")
        return True
    return False


def main():
    test_oracle_matches_sylvester_contract()
    print("PASS test_oracle_matches_sylvester_contract")
    test_non_power_of_two_is_refused()
    print("PASS test_non_power_of_two_is_refused")
    if _gpu_gate():
        return 0
    failures = 0
    impl = _impl()
    emu = _emu()
    for label, n, shape, dtype, scale, cos_lo, rel_hi in (
        # fp32: float32 addition-order noise vs the fp64 Sylvester oracle
        ("fp32/n128", 128, (17, 128), torch.float32, 128**-0.5, 0.999999, 1e-5),
        ("fp32/n512/3d", 512, (3, 9, 512), torch.float32, 512**-0.5, 0.999999, 1e-5),
        ("fp32/n4096", 4096, (2, 4096), torch.float32, 4096**-0.5, 0.999999, 1e-5),
        # bf16 io: quantization dominates (emu shows the identical error
        # band), so the oracle gate relaxes and a direct triton==emu gate
        # (both compute fp32 then round back) carries the parity claim
        ("bf16/n128", 128, (33, 128), torch.bfloat16, 1.0, 0.99999, 1e-2),
    ):
        cos, rel = _parity(n, shape, dtype, scale, impl._triton_hadamard_transform)
        ok = cos > cos_lo and rel < rel_hi
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} triton-parity[{label}] cos={cos:.9f} rel={rel:.2e}")
        x = torch.randn(shape, dtype=torch.float32, device="cuda").to(dtype)
        a = impl._triton_hadamard_transform(x, scale=scale)
        b = emu(x, scale=scale)
        if torch.equal(a, b):
            ok_x = True
            rel_x = 0.0
        else:
            rel_x = float((a.float() - b.float()).abs().max()) / (
                float(a.float().abs().max()) + 1e-30)
            ok_x = rel_x < 1e-6  # fp32 addition-order tail; bf16 is bit-exact
        failures += 0 if ok_x else 1
        print(f"{'PASS' if ok_x else 'FAIL'} triton-vs-emu[{label}] rel={rel_x:.2e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
