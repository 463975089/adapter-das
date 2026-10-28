"""Channel-wise int4 MoE kwarg rewrite (2026-10-25): regression pins.

box16 dsv41f-INT4 bring-up conviction: compressed-tensors pack-quantized
strategy="channel" (group=-1) arrives at the tree's triton runner with
block_shape=[0,-1]; invoke_fused_moe_kernel's gptq dequant branch only
fires for block_shape[1] > 0, so the u8-packed weights hit the plain
fused_moe_kernel's tl.dot(bf16, u8) and triton aborts at compile time
(first hit = decode CUDA-graph capture). channel == one group per whole
K row: the rewrite emits block_shape=[0, max(K_up, K_down)] so cdiv==1
on both expert GEMMs and the [E, N, 1] scales the scheme produced match.

Pinned here at the pure-logic layer (torch tensors, no sglang / no GPU):

  * predicate  — which block_shapes count as channel ([0,-1], None,
                 [0,0]) vs group-wise ([0,128] untouched);
  * group math — max(K_up, K_down) from packed shapes (K//2 -> K);
  * scales     — 2D [E, N] defensive lift to [E, N, 1]; 3D untouched;
  * idempotence / non-int4 passthrough at the wrapper level;
  * off-mode and REGISTRY <-> docs/env-vars.md reconciliation (L1 gate).

Run: python tests/adapter/test_wna16_channel_moe.py  (或 pytest)
"""

import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hcu_adapter import hcu_env, wna16_channel_moe  # noqa: E402


def test_channel_predicate_matches_only_nonpositive_group():
    # group-wise (positive) never matches — the stock path is correct there
    assert wna16_channel_moe._channel_group_fix(
        torch.zeros(2, 8, 4), torch.zeros(2, 4, 8), {"block_shape": [0, 128]}
    ) is False
    # channel sentinel from CompressedTensorsWNA16TritonMoE
    kw = {"block_shape": [0, -1]}
    assert wna16_channel_moe._channel_group_fix(
        torch.zeros(2, 8, 4), torch.zeros(2, 4, 8), kw
    ) is True
    assert kw["block_shape"][1] > 0
    # no block_shape at all (channel callers may omit it) -> rewrite
    assert wna16_channel_moe._channel_group_fix(
        torch.zeros(2, 8, 4), torch.zeros(2, 4, 8), {}
    ) is True
    # zero group is degenerate, not group-wise -> rewrite
    assert wna16_channel_moe._channel_group_fix(
        torch.zeros(2, 8, 4), torch.zeros(2, 4, 8), {"block_shape": [0, 0]}
    ) is True


def test_group_spans_both_gemms():
    # w13 packed [E, 2N, K//2] u8, w2 packed [E, H, I//2] u8; group must
    # be >= max(K, I) so cdiv(K_dim, group) == 1 on both invokes.
    w1 = torch.zeros(4, 64, 3584)   # K = 7168
    w2 = torch.zeros(4, 7168, 1024)  # I = 2048
    kw = {"block_shape": [0, -1]}
    wna16_channel_moe._channel_group_fix(w1, w2, kw)
    assert kw["block_shape"] == [0, 7168]
    # inter larger than hidden: group rides the down GEMM instead
    w1 = torch.zeros(4, 64, 512)    # K = 1024
    w2 = torch.zeros(4, 1024, 4096)  # I = 8192
    kw = {"block_shape": [0, -1]}
    wna16_channel_moe._channel_group_fix(w1, w2, kw)
    assert kw["block_shape"] == [0, 8192]


def test_scale_lift_2d_only():
    # scheme-transposed channel scales are already [E, N, 1]: untouched
    s = torch.ones(4, 64, 1)
    kw = {"block_shape": [0, -1], "w1_scale": s, "w2_scale": s}
    wna16_channel_moe._channel_group_fix(
        torch.zeros(4, 64, 8), torch.zeros(4, 8, 4), kw
    )
    assert kw["w1_scale"] is s and kw["w2_scale"] is s
    assert kw["w1_scale"].dim() == 3
    # defensive 2D [E, N] lift -> [E, N, 1] contiguous
    s2 = torch.ones(4, 64)
    kw = {"block_shape": [0, -1], "w1_scale": s2}
    wna16_channel_moe._channel_group_fix(
        torch.zeros(4, 64, 8), torch.zeros(4, 8, 4), kw
    )
    assert kw["w1_scale"].shape == (4, 64, 1)
    assert kw["w1_scale"].is_contiguous()
    # missing scales: rewrite still lands (kernel asserts its own None)
    kw = {"block_shape": [0, -1]}
    assert wna16_channel_moe._channel_group_fix(
        torch.zeros(4, 64, 8), torch.zeros(4, 8, 4), kw
    ) is True


def test_rewrite_idempotent_and_logged_once():
    wna16_channel_moe._REWRITE_LOG.clear()
    w1, w2 = torch.zeros(4, 64, 8), torch.zeros(4, 8, 4)
    kw = {"block_shape": [0, -1]}
    assert wna16_channel_moe._channel_group_fix(w1, w2, kw) is True
    first = dict(kw)
    # rewritten block_shape no longer matches the channel predicate
    assert wna16_channel_moe._channel_group_fix(w1, w2, kw) is False
    assert kw == first
    assert len(wna16_channel_moe._REWRITE_LOG) == 1


def test_wrapper_scopes_to_int4_face():
    calls = []

    def orig(hidden_states, w1, w2, topk_output, cfg, **kwargs):
        calls.append(kwargs)
        return "out"

    wrap = None
    # emulate install()'s wrapper construction without importing sglang
    src = pathlib.Path(wna16_channel_moe.__file__).read_text(encoding="utf-8")
    assert "def fused_experts_channel(" in src

    def fused_experts_channel(hidden_states, w1, w2, topk_output,
                              moe_runner_config, **kwargs):
        if kwargs.get("use_int4_w4a16"):
            wna16_channel_moe._channel_group_fix(w1, w2, kwargs)
        return orig(hidden_states, w1, w2, topk_output, moe_runner_config,
                    **kwargs)

    wrap = fused_experts_channel
    w1, w2 = torch.zeros(4, 64, 8), torch.zeros(4, 8, 4)
    # fp8 w8a8 face: untouched
    kw = {"use_fp8_w8a8": True, "block_shape": [128, 128]}
    wrap(None, w1, w2, None, None, **kw)
    assert calls[-1]["block_shape"] == [128, 128]
    # int4 group-wise face: untouched
    kw = {"use_int4_w4a16": True, "block_shape": [0, 128]}
    wrap(None, w1, w2, None, None, **kw)
    assert calls[-1]["block_shape"] == [0, 128]
    # int4 channel face: rewritten
    kw = {"use_int4_w4a16": True, "block_shape": [0, -1]}
    wrap(None, w1, w2, None, None, **kw)
    assert calls[-1]["block_shape"] == [0, 16]


def test_registry_entry_and_manual_reconcile():
    entry = hcu_env.REGISTRY.get("HCU_WNA16_CHANNEL_MOE")
    assert entry is not None, "REGISTRY key missing"
    assert entry["default"] == "1"
    manual = (ROOT / "docs" / "env-vars.md").read_text(encoding="utf-8")
    assert "| `HCU_WNA16_CHANNEL_MOE` |" in manual, "env-vars.md row missing"
    assert entry["about"].split("（")[0] in manual  # 头段与手册一致


def test_sglang_bridge_wiring():
    src = (ROOT / "hcu_adapter" / "bridges" / "sglang.py").read_text(
        encoding="utf-8"
    )
    assert "wna16_channel_moe as _wnch" in src
    assert "_wnch.install()" in src
    # chained after the other two fused_experts arms (10 -> 10.5 -> 10.6)
    assert src.index("aiter_moe as _amo") < src.index("w4a16_moe as _w16m")
    assert src.index("w4a16_moe as _w16m") < src.index("wna16_channel_moe as _wnch")


def test_env_resolution_default_on():
    os.environ.pop("HCU_WNA16_CHANNEL_MOE", None)
    v, src = hcu_env.resolve("HCU_WNA16_CHANNEL_MOE")
    assert (v or "1").strip() not in ("0", "off", "false")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
