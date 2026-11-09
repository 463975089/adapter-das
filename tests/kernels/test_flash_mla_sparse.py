# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""flash_mla_sparse_fwd emu numerical parity test (DSA decode sparse path).

Builds a reference: for each query row pick topk token ids, gather from a
paged pool, scalar FP64 attention with a separate virtual sink key,
and compare against `sgl_kernel.flash_mla_sparse.flash_mla_sparse_fwd`.
Also exercises -1 tail padding, topk_length clipping, and attn_sink.

Run: python tests/kernels/test_flash_mla_sparse.py
"""

import math
import sys

import torch

def _reference(q, kv_flat, idx, sm_scale, d_v, sink=None, topk_length=None):
    """Independent scalar dot products and an explicit zero-valued sink key."""
    s_q, H, d_qk = q.shape
    outs = torch.empty(s_q, H, d_v, dtype=torch.float64)
    lses = torch.empty(s_q, H, dtype=torch.float64)
    for t in range(s_q):
        limit = idx.shape[1] if topk_length is None else int(topk_length[t])
        # Retain duplicate selectors: they represent distinct softmax entries.
        keys = [kv_flat[int(token), 0].double().tolist()
                for token in idx[t, :limit] if int(token) >= 0]
        for head in range(H):
            query = q[t, head].double().tolist()
            logits = [math.fsum(a * b for a, b in zip(query, key)) * sm_scale
                      for key in keys]
            values = [key[:d_v] for key in keys]
            if sink is not None and math.isfinite(float(sink[head])):
                logits.append(float(sink[head]))
                values.append([0.0] * d_v)
            maximum = max(logits)
            weights = [math.exp(value - maximum) for value in logits]
            denominator = math.fsum(weights)
            lses[t, head] = maximum + math.log(denominator)
            outs[t, head] = torch.tensor([
                math.fsum(weight * value[channel]
                          for weight, value in zip(weights, values)) / denominator
                for channel in range(d_v)
            ], dtype=torch.float64)
    return outs, lses


def _metric(o, ref):
    """Dual metric per test-suite.md: cos (layout bugs destroy it) + maxrel."""
    of, rf = o.double().flatten(), ref.double().flatten()
    denom = of.norm() * rf.norm()
    cos = (of @ rf / denom).item() if denom > 0 else 0.0
    scale = rf.abs().max().item() or 1.0
    maxrel = ((of - rf).abs() / scale).max().item()
    return cos, maxrel


def main():
    from sgl_kernel.flash_mla_sparse import flash_mla_sparse_fwd

    dev = torch.device("cpu")  # pure-torch emu: identical math on CPU
    generator = torch.Generator().manual_seed(7)
    s_q, H, d_qk, d_v = 8, 3, 576, 512
    pool_tokens, topk = 300, 16
    kv = torch.randn(pool_tokens, 1, d_qk, dtype=torch.bfloat16, generator=generator)
    q = torch.randn(s_q, H, d_qk, dtype=torch.bfloat16, generator=generator)

    # random distinct indices; last two rows get -1 tails
    idx = torch.stack(
        [torch.randperm(pool_tokens, generator=generator)[:topk] for _ in range(s_q)]
    ).to(torch.int32)
    idx[-1, 8:] = -1
    idx[-2, 12:] = -1
    topk_length = torch.tensor([topk] * (s_q - 2) + [8, 12], dtype=torch.int32)

    scale = d_qk**-0.5
    sink = torch.randn(H, generator=generator) * 1e-3

    fails = 0
    for label, sink_on, tkl in (
        ("plain", None, None),
        ("sink", sink.clone(), None),
        ("sink-sentinel", torch.full((H,), -1e30), None),
        ("tailpad", None, topk_length.clone()),
        ("sink+tailpad", sink.clone(), topk_length.clone()),
    ):
        o, lse, _ = flash_mla_sparse_fwd(
            q=q.to(dev),
            kv=kv.to(dev),
            indices=idx.to(dev),
            sm_scale=scale,
            d_v=d_v,
            attn_sink=sink_on,
            topk_length=tkl,
        )
        ref, reference_lse = _reference(q, kv, idx, scale, d_v, sink_on, tkl)
        cos, maxrel = _metric(o, ref)
        lse_ok = torch.allclose(lse.double(), reference_lse, rtol=2e-6, atol=2e-6)
        ok = cos >= 0.9999 and maxrel <= 5e-2 and lse_ok
        status = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        print(
            f"[{status}] {label}: cos={cos:.6f} maxrel={maxrel:.2e} lse={lse_ok}"
        )
    print(f"flash_mla_sparse emu: {'5/5 PASS' if fails == 0 else f'{5-fails}/5 PASS'}")
    return 1 if fails else 0


def test_sparse_emu_contract():
    """CPU numerical contract; not evidence of HCU kernel execution."""
    assert main() == 0


def test_sparse_duplicate_selectors_and_virtual_sink():
    """Selectors are a multiset; padding and selector order preserve semantics."""
    from sgl_kernel.flash_mla_sparse import flash_mla_sparse_fwd

    q = torch.zeros(2, 2, 2, dtype=torch.bfloat16)
    kv = torch.tensor([[[2.0, 4.0]], [[6.0, 8.0]], [[100.0, -100.0]]],
                      dtype=torch.bfloat16)
    # The second row changes order. A valid but clipped selector must not count.
    idx = torch.tensor([[0, 1, 0, -1, 2], [0, 0, 1, -1, 2]], dtype=torch.int32)
    lengths = torch.tensor([3, 4], dtype=torch.int32)
    sink = torch.tensor([0.0, -torch.inf])
    expected = torch.tensor([[[2.5, 4.0], [10 / 3, 16 / 3]]] * 2,
                            dtype=torch.float64)
    expected_lse = torch.tensor([[math.log(4), math.log(3)]] * 2,
                                dtype=torch.float64)
    reference, reference_lse = _reference(q, kv, idx, 0.5, 2, sink, lengths)
    torch.testing.assert_close(reference, expected, rtol=0, atol=1e-15)
    torch.testing.assert_close(reference_lse, expected_lse, rtol=0, atol=1e-15)
    output, lse, _ = flash_mla_sparse_fwd(q, kv, idx, 0.5, 2, sink, lengths)
    torch.testing.assert_close(output, expected.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(lse.double(), expected_lse, rtol=2e-6, atol=2e-6)


def test_sparse_rope_channels_score_without_becoming_values():
    """Non-value query channels and the explicit scale affect softmax weights."""
    from sgl_kernel.flash_mla_sparse import flash_mla_sparse_fwd

    q = torch.tensor([[[0.0, 2.0]]], dtype=torch.bfloat16)
    kv = torch.tensor([[[2.0, 1.0]], [[6.0, -1.0]]], dtype=torch.bfloat16)
    idx = torch.tensor([[0, 1]], dtype=torch.int32)
    sink = torch.tensor([0.0])
    denominator = math.exp(1) + math.exp(-1) + 1
    expected = (2 * math.exp(1) + 6 * math.exp(-1)) / denominator
    reference, reference_lse = _reference(q, kv, idx, 0.5, 1, sink)
    assert math.isclose(float(reference[0, 0, 0]), expected, rel_tol=1e-15)
    output, lse, _ = flash_mla_sparse_fwd(q, kv, idx, 0.5, 1, sink)
    torch.testing.assert_close(output, reference.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(lse.double(), reference_lse, rtol=2e-6, atol=2e-6)


def test_sparse_emu_row_chunks_match_single_pass():
    """HCU_FLASHMLA_SPARSE_CHUNK_MB row split: bitwise equal to one pass."""
    import sgl_kernel.flash_mla_sparse as fms

    s_q, H, d_qk, d_v, pool_tokens, topk = 37, 4, 576, 512, 200, 32
    kv = torch.randn(pool_tokens, 1, d_qk, dtype=torch.bfloat16)
    q = torch.randn(s_q, H, d_qk, dtype=torch.bfloat16)
    idx = torch.stack(
        [torch.randperm(pool_tokens)[:topk] for _ in range(s_q)]
    ).to(torch.int32)
    idx[5, 20:] = -1
    tkl = torch.full((s_q,), topk, dtype=torch.int32)
    tkl[7] = 9
    sink = torch.randn(H) * 1e-3
    saved = fms._CHUNK_MB
    try:
        outs = []
        for budget in (1 << 20, 1):  # one pass vs ~3 rows per chunk
            fms._CHUNK_MB = budget
            outs.append(fms.flash_mla_sparse_fwd(
                q=q, kv=kv, indices=idx, sm_scale=d_qk**-0.5, d_v=d_v,
                attn_sink=sink, topk_length=tkl))
    finally:
        fms._CHUNK_MB = saved
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])


def test_sparse_emu_empty_rows():
    """s_q == 0 (idle DP rank / empty extend) returns empty o/lse."""
    from sgl_kernel.flash_mla_sparse import flash_mla_sparse_fwd

    o, lse, _ = flash_mla_sparse_fwd(
        q=torch.empty(0, 4, 576, dtype=torch.bfloat16),
        kv=torch.randn(64, 1, 576, dtype=torch.bfloat16),
        indices=torch.empty(0, 1, 16, dtype=torch.int32),
        sm_scale=576**-0.5, d_v=512)
    assert o.shape == (0, 4, 512) and o.dtype == torch.bfloat16
    assert lse.shape == (0, 4) and lse.dtype == torch.float32

if __name__ == "__main__":
    sys.exit(main())
