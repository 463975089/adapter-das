# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU routing contracts for automatic GDN AITER verify selection."""
import inspect
import math
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest import mock

import pytest

from hcu_adapter import gdn_aiter as bridge


DEV = types.SimpleNamespace(type="cuda", index=0)
TORCH = types.SimpleNamespace(bfloat16="bf16", float32="fp32", int32="i32", int64="i64")


class TensorMetadata:
    def __init__(self, shape, dtype="bf16", strides=None):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = DEV
        dense = []
        n = 1
        for dim in reversed(shape):
            dense.append(n)
            n *= dim
        self._dense = tuple(reversed(dense))
        self._strides = tuple(strides) if strides else self._dense

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]

    def is_contiguous(self):
        return self._strides == self._dense

    def numel(self):
        return math.prod(self.shape)

    def contiguous(self):
        return TensorMetadata(self.shape, self.dtype)


PARAMETERS = """A_log, a, dt_bias, softplus_beta, softplus_threshold,
q, k, v, b, initial_state_source, initial_state_indices,
scale=None, use_qk_l2norm_in_kernel=False, cu_seqlens=None, is_kda=False,
lower_bound=None, disable_state_update=False, intermediate_states_buffer=None,
intermediate_state_indices=None, cache_steps=None, retrieve_parent_token=None"""


def faces(extra=""):
    ns = {"calls": [], "error": None, "__name__": "cpu_gdn_face"}
    exec("def face(" + PARAMETERS + extra + "):\n"
         " calls.append(('upstream', dict(locals())))\n"
         " return 'upstream'\n", ns)
    vendor = {"calls": ns["calls"], "error": None, "__name__": "cpu_aiter_face"}
    exec("def vendor(" + PARAMETERS + "):\n"
         " calls.append(('aiter', dict(locals())))\n"
         " if error is not None: raise error\n"
         " return 'aiter'\n", vendor)
    return ns, vendor


def inputs():
    return dict(
        A_log=TensorMetadata((24,), "fp32"), a=TensorMetadata((1, 64, 24)),
        dt_bias=TensorMetadata((24,), "fp32"), softplus_beta=1.0,
        softplus_threshold=20.0,
        q=TensorMetadata((1, 64, 8, 128), strides=(327680, 5120, 128, 1)),
        k=TensorMetadata((1, 64, 8, 128), strides=(327680, 5120, 128, 1)),
        v=TensorMetadata((1, 64, 24, 128), strides=(327680, 5120, 128, 1)),
        b=TensorMetadata((1, 64, 24), strides=(3072, 48, 2)),
        initial_state_source=TensorMetadata((30, 24, 128, 128), "fp32"),
        initial_state_indices=TensorMetadata((16,), "i32"),
        cu_seqlens=TensorMetadata((17,), "i32"),
        disable_state_update=True,
        intermediate_states_buffer=TensorMetadata((18, 4, 24, 128, 128), "fp32"),
        intermediate_state_indices=TensorMetadata((16,), "i32"),
        cache_steps=4, retrieve_parent_token=TensorMetadata((16, 4), "i32"),
    )


def test_cached_alias_signature_and_snapshot_contract():
    ns, vendor = faces(", cache_ring=False, match_cutedsl_decode=False")
    fn = ns["face"]
    alias = fn
    signature = inspect.signature(fn)
    bridge._bind_target_verify_aiter(fn, vendor["vendor"], TORCH)
    assert alias is fn and inspect.signature(alias) == signature
    p = inputs()
    assert alias(**p) == "aiter"
    call = ns["calls"][-1][1]
    assert call["intermediate_states_buffer"] is p["intermediate_states_buffer"]
    assert call["initial_state_source"] is p["initial_state_source"]
    assert call["q"] is p["q"]  # token-strided projection view is preserved.
    assert call["b"].is_contiguous()
    assert call["b"] is not p["b"]
    assert "cache_ring" not in call


@pytest.mark.parametrize("change", [
    {"disable_state_update": False}, {"is_kda": True}, {"cache_ring": True},
    {"match_cutedsl_decode": True}, {"cache_steps": 3},
    {"initial_state_source": TensorMetadata((30, 24, 128, 128), "bf16")},
    {"q": TensorMetadata((1, 64, 8, 64))},
])
def test_unsupported_contract_keeps_upstream(change):
    ns, vendor = faces(", cache_ring=False, match_cutedsl_decode=False")
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], TORCH)
    p = inputs()
    p.update(change)
    assert fn(**p) == "upstream"
    assert [x[0] for x in ns["calls"]] == ["upstream"]


def test_provider_failure_is_not_retried_after_possible_write():
    ns, vendor = faces()
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], TORCH)
    vendor["error"] = RuntimeError("provider wrote then failed")
    with pytest.raises(RuntimeError, match="provider wrote"):
        fn(**inputs())
    assert [x[0] for x in ns["calls"]] == ["aiter"]


def test_unknown_upstream_feature_rejected_before_mutation():
    ns, vendor = faces(", new_snapshot_protocol=False")
    fn = ns["face"]
    code = fn.__code__
    with pytest.raises(RuntimeError, match="unreviewed"):
        bridge._bind_target_verify_aiter(fn, vendor["vendor"], TORCH)
    assert fn.__code__ is code and bridge._VERIFY_DISPATCH not in ns


def test_incompatible_vendor_rejected_before_mutation():
    ns, _ = faces()
    code = ns["face"].__code__
    with pytest.raises(RuntimeError, match="AITER verify contract"):
        bridge._bind_target_verify_aiter(ns["face"], lambda: None, TORCH)
    assert ns["face"].__code__ is code


def test_binding_is_idempotent():
    ns, vendor = faces()
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], TORCH)
    assert bridge._bind_target_verify_aiter(fn, vendor["vendor"], TORCH) is fn
    assert fn(**inputs()) == "aiter"
    assert len(ns["calls"]) == 1


@pytest.mark.parametrize("mode", ["auto", "triton"])
def test_unavailable_auto_and_explicit_rollback_do_not_import_vendor(mode):
    from hcu_adapter import hcu_env
    with mock.patch.object(hcu_env, "resolve", return_value=(mode, "test")), \
         mock.patch.object(bridge, "_verify_auto_stack_supported", return_value=False), \
         mock.patch.object(bridge, "_load_target_verify_providers", side_effect=AssertionError("must not import")), \
         mock.patch.object(bridge, "_bind_target_verify_aiter", side_effect=AssertionError("must not install")):
        bridge._install_target_verify_aiter()


def test_invalid_selector_is_loud():
    from hcu_adapter import hcu_env
    with mock.patch.object(hcu_env, "resolve", return_value=("bad", "test")):
        with pytest.raises(ValueError, match="must be"):
            bridge._install_target_verify_aiter()


def test_fresh_interpreter_cached_alias_route():
    env = dict(os.environ, HCU_ADAPTER_HOST="none", _HCU_ADAPTER_ACTIVE_HOST="none",
               HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    script = ("import runpy; m=runpy.run_path(" + repr(str(Path(__file__).resolve())) +
              "); m['test_cached_alias_signature_and_snapshot_contract'](); print('child-pass')")
    out = subprocess.check_output([sys.executable, "-c", script], env=env,
                                  timeout=60, text=True)
    assert "child-pass" in out


def test_registered_kill_switch_wins_over_explicit_aiter():
    from hcu_adapter import kill_switch
    assert "gdn_verify" in kill_switch.SWITCHES
    with mock.patch.dict(os.environ, {
        "HCU_DISABLE_OPS": "gdn_verify", "HCU_LINEAR_ATTN_GDN_VERIFY": "aiter",
    }), mock.patch.object(bridge, "_bind_target_verify_aiter", side_effect=AssertionError("disabled")):
        bridge._install_target_verify_aiter()


@pytest.mark.parametrize("strides", [None, (48, 2)])
def test_real_sglang_flattened_beta_contract(strides):
    ns, vendor = faces(", cache_ring=False, match_cutedsl_decode=False")
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], TORCH)
    p = inputs()
    p["a"] = TensorMetadata((64, 24))
    p["b"] = TensorMetadata((64, 24), strides=strides)
    assert fn(**p) == "aiter"
    call = ns["calls"][-1][1]
    assert call["b"].shape == (64, 24)
    assert call["b"].is_contiguous()


def test_malformed_beta_axes_retain_original_contract():
    ns, vendor = faces()
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], TORCH)
    p = inputs()
    p["b"] = TensorMetadata((32, 48))
    assert fn(**p) == "upstream"


def auto_torch(arch="gfx936:sramecc+:xnack-"):
    calls = []
    def properties(device):
        calls.append(device.index)
        return types.SimpleNamespace(gcnArchName=arch)
    torch = types.SimpleNamespace(**vars(TORCH),
        cuda=types.SimpleNamespace(get_device_properties=properties))
    return torch, calls


def auto_inputs():
    p = inputs()
    p["use_qk_l2norm_in_kernel"] = True
    return p


def test_auto_default_uses_vendor_and_caches_assigned_device_probe():
    ns, vendor = faces()
    torch, probes = auto_torch()
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], torch, policy="auto")
    assert fn(**auto_inputs()) == "aiter"
    assert fn(**auto_inputs()) == "aiter"
    assert probes == [0]
    assert getattr(fn, bridge._VERIFY_POLICY) == "auto"


@pytest.mark.parametrize("change", [
    {"use_qk_l2norm_in_kernel": False},
    {"softplus_beta": 2.0}, {"softplus_threshold": 16.0},
    {"lower_bound": -1.0}, {"scale": 0.5},
    {"a": TensorMetadata((1,64,24), "fp64")},
    {"A_log": TensorMetadata((24,), "bf16")},
    {"cu_seqlens": TensorMetadata((20,), "i32")},
    {"q": TensorMetadata((1,64,16,128)),
     "k": TensorMetadata((1,64,16,128))},
])
def test_auto_unvalidated_metadata_keeps_upstream_without_gpu_probe(change):
    ns, vendor = faces()
    torch, probes = auto_torch()
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], torch, policy="auto")
    p = auto_inputs(); p.update(change)
    assert fn(**p) == "upstream"
    assert not probes
    assert [x[0] for x in ns["calls"]] == ["upstream"]


@pytest.mark.parametrize("arch", ["gfx938", "gfx942", "unknown"])
def test_auto_unvalidated_arch_keeps_upstream(arch):
    ns, vendor = faces()
    torch, probes = auto_torch(arch)
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], torch, policy="auto")
    assert fn(**auto_inputs()) == "upstream"
    assert probes == [0]


def test_auto_queries_the_tensor_device_instead_of_device_zero():
    ns, vendor = faces()
    torch, probes = auto_torch()
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], torch, policy="auto")
    p = auto_inputs()
    device = types.SimpleNamespace(type="cuda", index=1)
    for value in p.values():
        if isinstance(value, TensorMetadata): value.device = device
    assert fn(**p) == "aiter"
    assert probes == [1]


def test_auto_device_probe_failure_retains_upstream():
    ns, vendor = faces()
    torch, _ = auto_torch()
    torch.cuda.get_device_properties = mock.Mock(side_effect=RuntimeError("no props"))
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], torch, policy="auto")
    assert fn(**auto_inputs()) == "upstream"


@pytest.mark.parametrize("missing", [None, "aiter", "triton", "torch", "sglang"])
def test_auto_stack_matches_validated_versions_only(missing):
    from importlib import metadata
    def version(name):
        if name == missing: return "unvalidated-build"
        return bridge._VERIFY_VALIDATED_STACK[name]
    with mock.patch.object(metadata, "version", side_effect=version):
        assert bridge._verify_auto_stack_supported() is (missing is None)


def test_auto_missing_package_is_upstream():
    from importlib import metadata
    with mock.patch.object(metadata, "version",
                           side_effect=metadata.PackageNotFoundError("aiter")):
        assert not bridge._verify_auto_stack_supported()


def test_default_installer_uses_auto_without_gpu_query():
    from hcu_adapter import registry
    ns, vendor = faces()
    torch, probes = auto_torch()
    with mock.patch.dict(os.environ, {"HCU_DISABLE_OPS": ""}), \
         mock.patch.object(bridge, "_verify_auto_stack_supported", return_value=True), \
         mock.patch.object(bridge, "_load_target_verify_providers",
                           return_value=(torch, ns["face"], vendor["vendor"])), \
         mock.patch.object(registry, "register") as register:
        with mock.patch.dict(os.environ, {}, clear=False):
            prior=os.environ.pop("HCU_LINEAR_ATTN_GDN_VERIFY", None)
            try: bridge._install_target_verify_aiter()
            finally:
                if prior is not None: os.environ["HCU_LINEAR_ATTN_GDN_VERIFY"]=prior
        assert not probes
        register.assert_called_once()
        assert ns["face"](**auto_inputs()) == "aiter"
        assert probes == [0]


def test_auto_abi_failure_retains_original_callable():
    from hcu_adapter import hcu_env
    ns, _ = faces()
    code = ns["face"].__code__
    torch, _ = auto_torch()
    with mock.patch.object(hcu_env, "resolve", return_value=("auto","default")), \
         mock.patch.object(bridge, "_verify_auto_stack_supported", return_value=True), \
         mock.patch.object(bridge, "_load_target_verify_providers",
                           return_value=(torch, ns["face"], lambda: None)):
        bridge._install_target_verify_aiter()
    assert ns["face"].__code__ is code
    assert bridge._VERIFY_DISPATCH not in ns


def test_explicit_aiter_preserves_manual_provider_selection():
    from hcu_adapter import hcu_env, registry
    ns, vendor = faces()
    torch, probes = auto_torch("gfx938")
    with mock.patch.object(hcu_env, "resolve", return_value=("aiter","explicit")), \
         mock.patch.object(bridge, "_verify_auto_stack_supported", return_value=False), \
         mock.patch.object(bridge, "_load_target_verify_providers",
                           return_value=(torch, ns["face"], vendor["vendor"])), \
         mock.patch.object(registry, "register"):
        bridge._install_target_verify_aiter()
    assert ns["face"](**inputs()) == "aiter"
    assert not probes


def test_kill_switch_wins_over_default_auto():
    with mock.patch.dict(os.environ, {"HCU_DISABLE_OPS": "gdn_verify"}), \
         mock.patch.object(bridge, "_load_target_verify_providers", side_effect=AssertionError("disabled")):
        bridge._install_target_verify_aiter()


def test_auto_vendor_error_never_retries_after_snapshot_write():
    ns, vendor = faces()
    torch, _ = auto_torch()
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], torch, policy="auto")
    vendor["error"] = RuntimeError("wrote snapshot")
    with pytest.raises(RuntimeError, match="wrote snapshot"): fn(**auto_inputs())
    assert [x[0] for x in ns["calls"]] == ["aiter"]


def test_gdn_entry_installs_verify_without_enabling_prefill_or_packed_decode():
    from hcu_adapter import hcu_env
    with mock.patch.object(bridge, "_install_target_verify_aiter") as verify, \
         mock.patch.object(hcu_env, "resolve", return_value=("auto", "default")), \
         mock.patch.object(bridge, "_install_fwd_h_aiter") as prefill, \
         mock.patch.object(bridge, "_install_packed_decode_aiter") as decode:
        bridge.install()
    verify.assert_called_once_with()
    prefill.assert_not_called()
    decode.assert_not_called()


def test_auto_policy_registered_as_default_with_explicit_rollback():
    from hcu_adapter import hcu_env
    from hcu_adapter.hcu_config import BACKEND_CATALOG
    assert hcu_env.REGISTRY["HCU_LINEAR_ATTN_GDN_VERIFY"]["default"] == "auto"
    assert set(BACKEND_CATALOG["linear_attn_gdn_verify"]["backends"]) == {
        "auto", "triton", "aiter",
    }


@pytest.mark.parametrize("dtype,provider", [("bf16", "aiter"), ("fp64", "upstream")])
def test_auto_model_dt_bias_dtype(dtype, provider):
    ns, vendor = faces()
    torch, _ = auto_torch()
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], torch, policy="auto")
    p = auto_inputs(); p["dt_bias"] = TensorMetadata((24,), dtype)
    assert fn(**p) == provider


def test_auto_linear_verify_without_parent_map_is_qualified():
    ns, vendor = faces()
    torch, _ = auto_torch()
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], torch, policy="auto")
    p=auto_inputs(); p["retrieve_parent_token"]=None
    assert fn(**p) == "aiter"


def test_auto_malformed_parent_map_keeps_upstream():
    ns, vendor = faces()
    torch, probes = auto_torch()
    fn = bridge._bind_target_verify_aiter(ns["face"], vendor["vendor"], torch, policy="auto")
    p=auto_inputs(); p["retrieve_parent_token"]=TensorMetadata((16,4), "fp32")
    assert fn(**p) == "upstream"
    assert not probes
