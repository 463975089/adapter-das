# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DeepGEMM's existing switches must reach the real installer.

CPU plumbing only: a clean interpreter with fake vendor/kernel providers
checks route selection. Actual kernel numerics live in tests/kernels.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("name", ["HCU_TRITON_MASKED", "HCU_TRITON_MASKED_FP8"])
@pytest.mark.parametrize("explicit,configured,expected,source", [
    (None, None, "1", "default"),
    ("0", None, "0", "explicit"),
    ("1", "0", "1", "explicit"),
    (None, "0", "0", "config"),
])
def test_grouped_gemm_switch_reaches_installer(name, explicit, configured, expected, source):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, HCU_ADAPTER_HOST="none", HIP_VISIBLE_DEVICES="",
               CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root))
    for key in ("HCU_TRITON_MASKED", "HCU_TRITON_MASKED_FP8", "HCU_CONFIG"):
        env.pop(key, None)
    if explicit is not None:
        env[name] = explicit
    script = r"""
import importlib.util
import sys
import types
import torch
from hcu_adapter import deep_gemm_dtk, hcu_config, hcu_env

name, configured, expected, source = sys.argv[1:]
hcu_config.lookup = lambda key: (configured if key == name and configured != "unset" else None)
assert hcu_env.resolve(name) == (expected, source), hcu_env.resolve(name)
assert hcu_env.enabled(name) == (expected == "1")

def vendor(*args, **kwargs):
    return "vendor"
def adapter(*args, **kwargs):
    return "adapter"

raw = types.ModuleType("deepgemm")
for key in ("m_grouped_bf16_gemm_nt_masked", "fp8_m_grouped_gemm_nt_masked",
            "m_grouped_fp8_gemm_nt_contiguous", "fp8_paged_mqa_logits",
            "get_paged_mqa_logits_metadata", "get_num_sms", "fp8_mqa_logits"):
    setattr(raw, key, vendor)
sys.modules["deepgemm"] = raw
# Only external providers are faked. The registry and installer are real.
for module_name, keys in {
    "deep_gemm_masked_triton": ["m_grouped_bf16_gemm_nt_masked"],
    "deep_gemm_contig_triton": ["m_grouped_bf16_gemm_nt_contiguous",
                              "get_mk_alignment_for_contiguous_layout",
                              "set_mk_alignment_for_contiguous_layout"],
    "deep_gemm_fp8_triton": ["fp8_m_grouped_gemm_nt_masked",
                           "m_grouped_fp8_gemm_nt_contiguous"],
}.items():
    module = types.ModuleType("hcu_adapter." + module_name)
    for key in keys:
        setattr(module, key, adapter)
    sys.modules[module.__name__] = module

for key in list(sys.modules):
    if key == "deep_gemm" or key.startswith("deep_gemm."):
        del sys.modules[key]
original_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda key, *a, **kw: (
    None if key == "deep_gemm" else original_find_spec(key, *a, **kw)
)
shim = deep_gemm_dtk.install()
assert shim is sys.modules["deep_gemm"]
enabled = expected == "1"
if name == "HCU_TRITON_MASKED":
    fn = shim.m_grouped_bf16_gemm_nt_masked
    assert (fn is adapter) == enabled
    # Even the retained vendor ABI wrapper is exercised when disabled.
    x = torch.ones(1, 1, 1, dtype=torch.bfloat16)
    assert fn(x, x, x, masked_m=torch.tensor([1]), expected_m=1) == (
        "adapter" if enabled else "vendor"
    )
else:
    for key in ("fp8_m_grouped_gemm_nt_masked", "m_grouped_fp8_gemm_nt_contiguous"):
        fn = getattr(shim, key)
        assert fn is (adapter if enabled else vendor)
        assert fn() == ("adapter" if enabled else "vendor")
    if enabled:
        value = object()
        assert shim.utils.layout.get_mn_major_tma_aligned_tensor(value) is value
    else:
        try:
            shim.utils.layout.get_mn_major_tma_aligned_tensor(object())
        except NotImplementedError:
            pass
        else:
            raise AssertionError("disabled FP8 must retain the unverified-layout refusal")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, name, configured or "unset", expected, source],
        env=env, cwd=root, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
