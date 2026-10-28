"""Regression: the slimquant loader shim must stay format-gated.

Commit 2685ffc root cause: `_SlimquantShard` dequantized every '.weight'
that had a '_scale' sibling for ALL checkpoints, not just slimquant_w4a8
ones. A compressed-tensors int-quantized checkpoint (weight int8 + scale
bf16, both consumed raw by the CT scheme) then loaded garbage: the
dequantized bf16 values were copy_'d into int8 params, truncating to all
zeros -> zero logits -> uniform token salad. FP8 checkpoints escaped only
because their scales are named 'weight_scale_inv'.

These tests pin the gate at the safetensors-handle level (CPU only, no
sglang import needed).
"""

import json

import torch
from safetensors import safe_open as real_safe_open
from safetensors.torch import save_file

from hcu_adapter.slimquant import _wrap_safe_open

Q_KEY = "model.layers.0.self_attn.q_proj.weight"
S_KEY = Q_KEY + "_scale"
NORM_KEY = "model.layers.0.input_layernorm.weight"
EXP_W_KEY = "model.layers.0.mlp.experts.3.down_proj.weight"
EXP_S_KEY = EXP_W_KEY + "_scale"

_W = torch.tensor([[127, -128, 3, 0]], dtype=torch.int8)
_S = torch.tensor([[0.03125]], dtype=torch.bfloat16)
_NORM = torch.linspace(-1, 1, 4, dtype=torch.bfloat16).unsqueeze(0)
_EXP_W = torch.tensor([[13, -14, 15, -16], [1, 2, 3, 4]], dtype=torch.int8)
_EXP_S = torch.tensor([[0.5], [0.25]], dtype=torch.bfloat16)


def _make_ckpt(root, quant_config):
    (root / "shard").mkdir(parents=True, exist_ok=True)
    cfg = {"architectures": ["FakeForCausalLM"]}
    if quant_config is not None:
        cfg["quantization_config"] = quant_config
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file(
        {
            Q_KEY: _W.clone(),
            S_KEY: _S.clone(),
            NORM_KEY: _NORM.clone(),
            EXP_W_KEY: _EXP_W.clone(),
            EXP_S_KEY: _EXP_S.clone(),
        },
        str(root / "shard" / "model-00001.safetensors"),
    )
    return root / "shard" / "model-00001.safetensors"


def _open(path):
    shim = _wrap_safe_open(real_safe_open)
    f = shim(str(path), framework="pt", device="cpu")
    return f


def _assert_passthrough(f):
    """CT-family contract: raw int8 weight AND its scale reach the loader."""
    assert S_KEY in f.keys(), "weight_scale filtered from keys() — would zero the layer"
    assert EXP_S_KEY in f.keys(), "expert scale filtered from keys()"
    w = f.get_tensor(Q_KEY)
    assert w.dtype == torch.int8 and torch.equal(w, _W), (
        f"weight transformed: {w.dtype} {w}"
    )
    assert torch.equal(f.get_tensor(S_KEY), _S), "scale value altered"
    assert torch.equal(f.get_tensor(NORM_KEY), _NORM), "norm tensor altered"
    assert torch.equal(f.get_tensor(EXP_W_KEY), _EXP_W), "expert packing changed"


def _assert_transformed(f):
    """slimquant contract: non-expert weight dequantized to bf16, its scale
    hidden from keys(); routed-expert int4 packing passes through and the
    stored scale is returned x16 (kernel convention)."""
    assert S_KEY not in f.keys()
    assert EXP_S_KEY in f.keys()
    w = f.get_tensor(Q_KEY)
    assert w.dtype == torch.bfloat16, f"expected bf16 dequant, got {w.dtype}"
    ref = (_W.float() * _S.float()).to(torch.bfloat16)
    assert torch.equal(w, ref)
    assert torch.equal(f.get_tensor(NORM_KEY), _NORM), "norm tensor altered"
    ew = f.get_tensor(EXP_W_KEY)
    assert ew.dtype == torch.int8 and torch.equal(ew, _EXP_W), (
        "expert packing changed"
    )
    assert torch.equal(f.get_tensor(EXP_S_KEY), (_EXP_S.float() * 16.0).to(torch.bfloat16))


def test_ct_int8_checkpoint_passthrough(tmp_path):
    path = _make_ckpt(tmp_path / "ct", {"format": "int-quantized"})
    _assert_passthrough(_open(path))


def test_ct_float_checkpoint_passthrough(tmp_path):
    path = _make_ckpt(tmp_path / "ctf", {"format": "float-quantized"})
    _assert_passthrough(_open(path))


def test_slimquant_checkpoint_transforms(tmp_path):
    path = _make_ckpt(tmp_path / "sq", {"format": "slimquant_w4a8"})
    _assert_transformed(_open(path))


def test_slimquant_method_field_transforms(tmp_path):
    # GLM-5 family exports carry quantization_config.quant_method instead
    # of .format — the gate must accept both spellings
    path = _make_ckpt(tmp_path / "sqm", {"quant_method": "slimquant_w4a8"})
    _assert_transformed(_open(path))


def test_no_quant_config_passthrough(tmp_path):
    # config present but declares nothing: determinate passthrough. The
    # catastrophic default (dequant) is exactly the commit-2685ffc bug — a
    # bare-w8a8 checkpoint would load all-zero int8 params silently.
    path = _make_ckpt(tmp_path / "nq", {})
    _assert_passthrough(_open(path))


def test_missing_config_passthrough(tmp_path):
    # determinate lookups never transform; only an unreadable config
    # (IO exception) preserves the legacy transform behavior
    path = _make_ckpt(tmp_path / "nc", None)
    (path.parent.parent / "config.json").unlink()
    _assert_passthrough(_open(path))


def test_nested_shard_dir_config_lookup(tmp_path):
    # checkpoint root one level above a nested shard subdir (real HF layout
    # margin: config sits beside the shards; allow one nesting level)
    root = tmp_path / "nested"
    (root / "shard").mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"quantization_config": {"format": "int-quantized"}}),
        encoding="utf-8",
    )
    deep = root / "shard" / "sub"
    deep.mkdir(parents=True, exist_ok=True)
    save_file({Q_KEY: _W.clone(), S_KEY: _S.clone()}, str(deep / "m.safetensors"))
    f = _open(deep / "m.safetensors")
    assert S_KEY in f.keys(), "scale filtered despite int-quantized config"
    assert f.get_tensor(Q_KEY).dtype == torch.int8, "weight transformed"


def test_orphan_shard_passthrough(tmp_path):
    # a shard with no reachable config.json: origin unknowable — the shim
    # must NOT transform (misfire = all-zero int8 weights, the 2685ffc bug)
    d = tmp_path / "orphan"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "m.safetensors"
    save_file({Q_KEY: _W.clone(), S_KEY: _S.clone()}, str(p))
    f = _open(p)
    assert S_KEY in f.keys(), "scale filtered on config-less shard"
    assert f.get_tensor(Q_KEY).dtype == torch.int8, "weight transformed"


def test_gate_cache_is_per_path(tmp_path):
    # one process, both kinds: the per-path cache must not cross-contaminate
    ct = _make_ckpt(tmp_path / "c2", {"format": "int-quantized"})
    sq = _make_ckpt(tmp_path / "s2", {"format": "slimquant_w4a8"})
    _assert_passthrough(_open(ct))
    _assert_transformed(_open(sq))
    _assert_passthrough(_open(ct))


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_fail = 0
    for fn in tests:
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(Path(td))
                print(f"PASS {fn.__name__}")
            except AssertionError as e:
                n_fail += 1
                print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if n_fail else 0)
