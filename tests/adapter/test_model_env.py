# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""R-08 B-tier model-aware env defaults: rule pin (no GPU / no sglang).

apply_model_env(model_config) must:
* set both DSA fusion escape hatches for a config exposing index_kpool
  (GLM-5.3-Flash / DeepSeek V3.2 family), leave them untouched otherwise;
* set SGLANG_MAMBA_CONV_DTYPE=float16 only for fp16 configs with mamba/GDN
  keys (bf16 GDN and fp16 dense must NOT get it);
* never clobber an explicit export;
* return exactly the applied lines.

apply_from_argv (primary serving arm, 2026-10-18 zero-env gate lesson) must:
* parse both --model-path VALUE and --model-path=VALUE forms;
* find DSA hint keys nested in composite configs (glm5_next text_config);
* leave explicit exports alone; no-op without --model-path or a bad path.

Pytest-collectable (L1 kernels segment collects tests/adapter); direct run
also works: python tests/adapter/test_model_env.py
(2026-10-19: module-level sys.exit crashed pytest collection -> INTERNALERROR,
refactored to functions + __main__ guard, same 15 checks).
"""

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter.model_env import (  # noqa: E402
    _model_path_from_argv,
    apply_from_argv,
    apply_model_env,
)

VARS = (
    "SGLANG_DSA_FUSE_TOPK",
    "SGLANG_DISABLE_DSA_INDEXER_FUSION",
    "SGLANG_MAMBA_CONV_DTYPE",
)


class _MC:  # ModelConfig stand-in: .dtype + .hf_config
    def __init__(self, dtype_name, **hf):
        self.dtype = types.SimpleNamespace(name=dtype_name)
        self.hf_config = types.SimpleNamespace(**hf)


class _Env:
    """save VARS on enter, restore on exit -- order-independent tests."""

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in VARS}
        for k in VARS:
            os.environ.pop(k, None)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cfg(td, **obj):
    json.dump(obj, open(os.path.join(td, "config.json"), "w"))


def test_apply_model_env():
    # 1. DSA model (GLM-5.3-Flash shape: index_kpool present)
    with _Env():
        applied = apply_model_env(_MC("bfloat16", index_kpool=4))
        assert os.environ.get("SGLANG_DSA_FUSE_TOPK") == "0", (
            f"dsa-sets-both: {applied}")
        assert os.environ.get("SGLANG_DISABLE_DSA_INDEXER_FUSION") == "1", (
            f"dsa-sets-both: {applied}")
        assert "SGLANG_MAMBA_CONV_DTYPE" not in os.environ, "dsa-no-conv-dtype"

    # 2. non-DSA model leaves everything alone
    with _Env():
        applied = apply_model_env(_MC("bfloat16", hidden_size=4096))
        assert applied == [] and all(k not in os.environ for k in VARS), (
            f"dense-untouched: {applied}")

    # 3. fp16 GDN bottom model gets the conv dtype
    with _Env():
        apply_model_env(_MC("float16", mamba_num_heads=8, hidden_size=2048))
        assert os.environ.get("SGLANG_MAMBA_CONV_DTYPE") == "float16", (
            "fp16-gdn-conv")

    # 4. bf16 GDN must NOT get it (dtype gate)
    with _Env():
        apply_model_env(_MC("bfloat16", mamba_num_heads=8))
        assert "SGLANG_MAMBA_CONV_DTYPE" not in os.environ, "bf16-gdn-no-conv"

    # 5. explicit export wins
    with _Env():
        for k, v in zip(VARS, ("1", "0", "bfloat16")):
            os.environ[k] = v
        applied = apply_model_env(_MC("float16", index_kpool=4,
                                      mamba_num_heads=8))
        assert applied == [], f"explicit-wins: {applied}"
        assert os.environ["SGLANG_DSA_FUSE_TOPK"] == "1", "explicit-wins"
        assert os.environ["SGLANG_DISABLE_DSA_INDEXER_FUSION"] == "0", (
            "explicit-wins")
        assert os.environ["SGLANG_MAMBA_CONV_DTYPE"] == "bfloat16", (
            "explicit-wins")


def test_apply_from_argv():
    assert _model_path_from_argv(
        ["x", "--model-path", "/m", "--tp-size", "4"]) == "/m", (
        "argv-parse-space-form")
    assert _model_path_from_argv(["x", "--model-path=/m2"]) == "/m2", (
        "argv-parse-eq-form")
    assert _model_path_from_argv(["x", "--tp", "4"]) is None, (
        "argv-parse-absent")

    # composite checkpoint: DSA keys live under text_config (glm5_next)
    with _Env(), tempfile.TemporaryDirectory() as td:
        _cfg(td, model_type="glm5_next",
             text_config={"dtype": "bfloat16", "index_kpool": 4,
                          "index_head_dim": 128})
        applied = apply_from_argv(["x", "--model-path", td])
        assert os.environ.get("SGLANG_DSA_FUSE_TOPK") == "0", (
            f"argv-nested-dsa: {applied}")
        assert os.environ.get("SGLANG_DISABLE_DSA_INDEXER_FUSION") == "1", (
            f"argv-nested-dsa: {applied}")
        assert "SGLANG_MAMBA_CONV_DTYPE" not in os.environ, "argv-nested-dsa"

    # explicit export wins over argv sniff
    with _Env():
        os.environ["SGLANG_DSA_FUSE_TOPK"] = "1"
        with tempfile.TemporaryDirectory() as td:
            _cfg(td, index_kpool=4)
            applied = apply_from_argv(["x", "--model-path", td])
        assert applied == ["SGLANG_DISABLE_DSA_INDEXER_FUSION=1"], (
            f"argv-explicit-wins: {applied}")
        assert os.environ["SGLANG_DSA_FUSE_TOPK"] == "1", "argv-explicit-wins"

    # fp16 GDN section gets conv dtype; bf16 section does not
    with _Env(), tempfile.TemporaryDirectory() as td:
        _cfg(td, dtype="float16", mamba_num_heads=8)
        apply_from_argv(["x", "--model-path", td])
        assert os.environ.get("SGLANG_MAMBA_CONV_DTYPE") == "float16", (
            "argv-fp16-gdn-conv")
    with _Env(), tempfile.TemporaryDirectory() as td:
        _cfg(td, dtype="bfloat16", mamba_num_heads=8)
        apply_from_argv(["x", "--model-path", td])
        assert "SGLANG_MAMBA_CONV_DTYPE" not in os.environ, (
            "argv-bf16-gdn-no-conv")

    # no argv / bad path -> no-op, no crash
    with _Env():
        assert apply_from_argv(["x"]) == [], "argv-absent-noop"
        assert apply_from_argv(
            ["x", "--model-path", "/nonexistent-model"]) == [], (
            "argv-badpath-noop")


if __name__ == "__main__":
    fails = 0
    for fn in (test_apply_model_env, test_apply_from_argv):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    print("RESULT:", "PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)
