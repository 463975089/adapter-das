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

Run: python tests/adapter/test_model_env.py
"""

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sgl_kernel._adapters.model_env import (  # noqa: E402
    _model_path_from_argv,
    apply_from_argv,
    apply_model_env,
)

FAILS = []
VARS = (
    "SGLANG_DSA_FUSE_TOPK",
    "SGLANG_DISABLE_DSA_INDEXER_FUSION",
    "SGLANG_MAMBA_CONV_DTYPE",
)


def ok(tag, cond, detail=""):
    print(f"{tag}: {'OK' if cond else 'FAIL ' + detail}", flush=True)
    if not cond:
        FAILS.append(tag)


class _MC:  # ModelConfig stand-in: .dtype + .hf_config
    def __init__(self, dtype_name, **hf):
        self.dtype = types.SimpleNamespace(name=dtype_name)
        self.hf_config = types.SimpleNamespace(**hf)


saved = {k: os.environ.get(k) for k in VARS}
try:
    for k in VARS:
        os.environ.pop(k, None)
    # 1. DSA model (GLM-5.3-Flash shape: index_kpool present)
    applied = apply_model_env(_MC("bfloat16", index_kpool=4))
    ok(
        "dsa-sets-both",
        os.environ.get("SGLANG_DSA_FUSE_TOPK") == "0"
        and os.environ.get("SGLANG_DISABLE_DSA_INDEXER_FUSION") == "1",
        str(applied),
    )
    ok("dsa-no-conv-dtype", "SGLANG_MAMBA_CONV_DTYPE" not in os.environ)

    # 2. non-DSA model leaves everything alone
    for k in VARS:
        os.environ.pop(k, None)
    applied = apply_model_env(_MC("bfloat16", hidden_size=4096))
    ok("dense-untouched", applied == [] and all(k not in os.environ for k in VARS))

    # 3. fp16 GDN bottom model gets the conv dtype
    for k in VARS:
        os.environ.pop(k, None)
    apply_model_env(_MC("float16", mamba_num_heads=8, hidden_size=2048))
    ok(
        "fp16-gdn-conv",
        os.environ.get("SGLANG_MAMBA_CONV_DTYPE") == "float16",
    )

    # 4. bf16 GDN must NOT get it (dtype gate)
    for k in VARS:
        os.environ.pop(k, None)
    apply_model_env(_MC("bfloat16", mamba_num_heads=8))
    ok("bf16-gdn-no-conv", "SGLANG_MAMBA_CONV_DTYPE" not in os.environ)

    # 5. explicit export wins
    for k, v in zip(VARS, ("1", "0", "bfloat16")):
        os.environ[k] = v
    applied = apply_model_env(_MC("float16", index_kpool=4, mamba_num_heads=8))
    ok(
        "explicit-wins",
        applied == []
        and os.environ["SGLANG_DSA_FUSE_TOPK"] == "1"
        and os.environ["SGLANG_DISABLE_DSA_INDEXER_FUSION"] == "0"
        and os.environ["SGLANG_MAMBA_CONV_DTYPE"] == "bfloat16",
        str(applied),
    )
finally:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

# --- argv arm (apply_from_argv) ---
saved = {k: os.environ.get(k) for k in VARS}
try:
    for k in VARS:
        os.environ.pop(k, None)

    ok(
        "argv-parse-space-form",
        _model_path_from_argv(["x", "--model-path", "/m", "--tp-size", "4"])
        == "/m",
    )
    ok(
        "argv-parse-eq-form",
        _model_path_from_argv(["x", "--model-path=/m2"]) == "/m2",
    )
    ok("argv-parse-absent", _model_path_from_argv(["x", "--tp", "4"]) is None)

    # composite checkpoint: DSA keys live under text_config (glm5_next)
    with tempfile.TemporaryDirectory() as td:
        json.dump(
            {
                "model_type": "glm5_next",
                "text_config": {
                    "dtype": "bfloat16",
                    "index_kpool": 4,
                    "index_head_dim": 128,
                },
            },
            open(os.path.join(td, "config.json"), "w"),
        )
        applied = apply_from_argv(["x", "--model-path", td])
        ok(
            "argv-nested-dsa",
            os.environ.get("SGLANG_DSA_FUSE_TOPK") == "0"
            and os.environ.get("SGLANG_DISABLE_DSA_INDEXER_FUSION") == "1"
            and "SGLANG_MAMBA_CONV_DTYPE" not in os.environ,
            str(applied),
        )

    # explicit export wins over argv sniff
    for k in VARS:
        os.environ.pop(k, None)
    os.environ["SGLANG_DSA_FUSE_TOPK"] = "1"
    with tempfile.TemporaryDirectory() as td:
        json.dump({"index_kpool": 4}, open(os.path.join(td, "config.json"), "w"))
        applied = apply_from_argv(["x", "--model-path", td])
        ok(
            "argv-explicit-wins",
            applied == ["SGLANG_DISABLE_DSA_INDEXER_FUSION=1"]
            and os.environ["SGLANG_DSA_FUSE_TOPK"] == "1",
            str(applied),
        )

    # fp16 GDN section gets conv dtype; bf16 section does not
    for k in VARS:
        os.environ.pop(k, None)
    with tempfile.TemporaryDirectory() as td:
        json.dump(
            {"dtype": "float16", "mamba_num_heads": 8},
            open(os.path.join(td, "config.json"), "w"),
        )
        apply_from_argv(["x", "--model-path", td])
        ok(
            "argv-fp16-gdn-conv",
            os.environ.get("SGLANG_MAMBA_CONV_DTYPE") == "float16",
        )
    for k in VARS:
        os.environ.pop(k, None)
    with tempfile.TemporaryDirectory() as td:
        json.dump(
            {"dtype": "bfloat16", "mamba_num_heads": 8},
            open(os.path.join(td, "config.json"), "w"),
        )
        apply_from_argv(["x", "--model-path", td])
        ok("argv-bf16-gdn-no-conv", "SGLANG_MAMBA_CONV_DTYPE" not in os.environ)

    # no argv / bad path -> no-op, no crash
    for k in VARS:
        os.environ.pop(k, None)
    ok("argv-absent-noop", apply_from_argv(["x"]) == [])
    ok(
        "argv-badpath-noop",
        apply_from_argv(["x", "--model-path", "/nonexistent-model"]) == [],
    )
finally:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
sys.exit(1 if FAILS else 0)
