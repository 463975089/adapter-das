# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""V4.1 cache defaults through the process-local platform configuration seam."""

import logging
import multiprocessing
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from hcu_adapter.features.dsv41 import fp4_indexer
from hcu_adapter.infra import hcu_config
from hcu_adapter.bridges import vllm as bridge
from hcu_adapter.infra.extensions import ExtensionReport, ExtensionStatus


def config(dtype="auto", model_type="deepseek_v41"):
    hf = NS(model_type=model_type, compress_ratios=[0, 1, 2], index_head_dim=128,
            index_n_heads=32, qk_rope_head_dim=64, kv_source_layer_ids=[1],
            index_source_layer_ids=[1])
    return NS(
        attention_config=NS(indexer_kv_dtype=dtype, indexer_sparse_logits=False),
        model_config=NS(hf_text_config=hf, enforce_eager=True),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           prefill_context_parallel_size=1, decode_context_parallel_size=1,
                           data_parallel_size=1, enable_dbo=False, enable_eplb=False),
        cache_config=NS(block_size=64),
        kernel_config=NS(monokernel=False, moe_backend="deep_gemm"),
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        use_v2_model_runner=True,
    )


def reports():
    return tuple(
        ExtensionReport(capability, capability, ExtensionStatus.ACTIVE,
                        frozenset({capability}), "CPU provider")
        for capability in sorted(bridge._FP4_INDEXER_CAPABILITIES)
    )


def publish(owners):
    for report in owners:
        if report.status is ExtensionStatus.ACTIVE:
            bridge._RUNTIME_ACTIONS.record_extension_health(report, lambda: None)
    bridge._RUNTIME_ACTIONS.record_extension_reports(owners)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    monkeypatch.setattr(hcu_config, "_STATE", {
        "loaded": True, "path": None, "ops": {}, "overrides": {}, "hosts": {},
    })
    for name in ("HCU_VLLM_FP4_INDEXER_IMPL", "HCU_VLLM_FP4_PACKER_IMPL",
                 "HCU_VLLM_SPARSE_INDEXER_IMPL", "HCU_VLLM_PCP_IMPL", "HCU_DISABLE_OPS"):
        monkeypatch.delenv(name, raising=False)
    publish(reports())


@pytest.mark.parametrize("model_type", ["deepseek_v41", "deepseek_v41_text"])
def test_auto_resolves_once_after_provider_health(model_type, caplog):
    cfg = config(model_type=model_type)
    assert fp4_indexer.default_requested(cfg)
    with caplog.at_level(logging.INFO, logger="hcu.adapter"):
        bridge._configure_indexer_kv_dtype(cfg)
        bridge._configure_indexer_kv_dtype(cfg)
    assert cfg.attention_config.indexer_kv_dtype == "mxfp4"
    assert sum("auto -> mxfp4" in record.message for record in caplog.records) == 1


@pytest.mark.parametrize("field,error", [
    ("tensor_parallel_size", "tensor_parallel_size=1 required"),
    ("data_parallel_size", "DP1 or DP8 required"),
])
def test_unsupported_parallelism_keeps_auto_and_rejects_explicit_mxfp4(field, error):
    cfg = config()
    setattr(cfg.parallel_config, field, 2)
    bridge._configure_indexer_kv_dtype(cfg)
    assert cfg.attention_config.indexer_kv_dtype == "auto"

    cfg.attention_config.indexer_kv_dtype = "mxfp4"
    bridge._configure_indexer_kv_dtype(cfg)
    assert cfg.attention_config.indexer_kv_dtype == "mxfp4"
    with pytest.raises(ValueError, match=error):
        fp4_indexer.validate_config(cfg)


@pytest.mark.parametrize("kv_transfer", [False, True])
def test_pcp8_default_obeys_kv_transfer_admission(monkeypatch, kv_transfer):
    monkeypatch.setenv("HCU_VLLM_PCP_IMPL", "native")
    cfg = config()
    cfg.parallel_config.prefill_context_parallel_size = 8
    cfg.parallel_config.enable_expert_parallel = True
    cfg.parallel_config.all2all_backend = "deepep_high_throughput"
    if kv_transfer:
        cfg.kv_transfer_config = NS()

    bridge._configure_indexer_kv_dtype(cfg)

    assert cfg.attention_config.indexer_kv_dtype == ("auto" if kv_transfer else "mxfp4")


@pytest.mark.parametrize("dtype", ["fp8", "mxfp4", "bf16", "invalid"])
def test_explicit_dtype_keeps_its_original_validation(dtype):
    cfg = config(dtype)
    bridge._configure_indexer_kv_dtype(cfg)
    assert cfg.attention_config.indexer_kv_dtype == dtype


@pytest.mark.parametrize("model_type", [
    "deepseek_v4", "deepseek_v3", "deepseek_v41_other", "minimax_m3", None,
])
def test_other_models_keep_auto(model_type):
    cfg = config(model_type=model_type)
    bridge._configure_indexer_kv_dtype(cfg)
    assert cfg.attention_config.indexer_kv_dtype == "auto"


@pytest.mark.parametrize("cfg", [None, NS(), NS(model_config=None)])
def test_missing_model_config_defers(cfg):
    assert not fp4_indexer.default_requested(cfg)
    bridge._configure_indexer_kv_dtype(cfg)


def test_default_predicate_import_and_call_are_framework_free():
    program = """
import builtins
import sys
from types import SimpleNamespace as NS
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'vllm', 'sglang', 'lightop', 'triton'}:
        raise AssertionError('framework import ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from hcu_adapter.features.dsv41 import fp4_indexer
from hcu_adapter.infra import hcu_config
hcu_config._STATE = {'loaded': True, 'path': None, 'ops': {},
                     'overrides': {}, 'hosts': {}}
cfg = NS(attention_config=NS(indexer_kv_dtype='auto'),
         model_config=NS(hf_text_config=NS(model_type='deepseek_v41')))
assert fp4_indexer.default_requested(cfg)
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", program,
                    str(Path(__file__).resolve().parents[2])],
                   check=True, timeout=20, capture_output=True, text=True)


@pytest.mark.parametrize("name,value", [
    ("HCU_VLLM_FP4_INDEXER_IMPL", "stock"),
    ("HCU_VLLM_FP4_PACKER_IMPL", "stock"),
    ("HCU_DISABLE_OPS", "vllm_fp4_indexer"),
    ("HCU_DISABLE_OPS", "vllm_fp4_packer"),
])
def test_explicit_opt_out_keeps_auto(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    cfg = config()
    bridge._configure_indexer_kv_dtype(cfg)
    assert cfg.attention_config.indexer_kv_dtype == "auto"


def test_config_file_opt_out_keeps_auto():
    hcu_config._STATE["overrides"]["HCU_VLLM_FP4_INDEXER_IMPL"] = "stock"
    cfg = config()
    bridge._configure_indexer_kv_dtype(cfg)
    assert cfg.attention_config.indexer_kv_dtype == "auto"


@pytest.mark.parametrize("owner_count", [0, 1])
def test_missing_provider_does_not_select_fp4(monkeypatch, owner_count):
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    publish(reports()[:owner_count])
    cfg = config()
    bridge._configure_indexer_kv_dtype(cfg)
    assert cfg.attention_config.indexer_kv_dtype == "auto"


def test_failed_report_does_not_select_fp4(monkeypatch):
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    first, second = reports()
    failed = ExtensionReport(second.entry_point_name, second.extension_name,
                             ExtensionStatus.FAILED, second.capabilities, "unavailable")
    publish((first, failed))
    cfg = config()
    bridge._configure_indexer_kv_dtype(cfg)
    assert cfg.attention_config.indexer_kv_dtype == "auto"


def test_unhealthy_owner_refuses_before_dtype_changes(monkeypatch):
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    first, second = reports()

    def failed():
        raise RuntimeError("provider lost its binding")

    bridge._RUNTIME_ACTIONS.record_extension_health(first, failed)
    bridge._RUNTIME_ACTIONS.record_extension_health(second, lambda: None)
    bridge._RUNTIME_ACTIONS.record_extension_reports((first, second))
    cfg = config()
    with pytest.raises(bridge.LatchedVllmRuntimeError, match="provider lost"):
        bridge._configure_indexer_kv_dtype(cfg)
    assert cfg.attention_config.indexer_kv_dtype == "auto"


def test_real_platform_passes_effective_dtype_to_core(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.platforms.rocm import RocmPlatform
    from hcu_adapter.bridges import vllm_moe_aiter

    cfg = config()
    cfg.parallel_config.worker_cls = "custom.Worker"
    cfg.parallel_config.distributed_executor_backend = "mp"
    seen = []
    monkeypatch.setattr(bridge, "require_runtime_healthy", lambda: None)
    monkeypatch.setattr(vllm_moe_aiter, "configure_padding_policy", lambda _: None)
    monkeypatch.setattr(RocmPlatform, "check_and_update_config",
                        classmethod(lambda cls, value: seen.append(
                            value.attention_config.indexer_kv_dtype)))
    bridge.HcuPlatform.check_and_update_config(cfg)
    assert seen == ["mxfp4"]
    assert cfg.parallel_config.worker_cls == "custom.Worker"


def test_real_attention_config_hash_and_serialization_use_effective_dtype():
    core = pytest.importorskip("vllm.config")
    from pydantic import TypeAdapter

    cfg = config()
    cfg.attention_config = core.AttentionConfig()
    original_hash = cfg.attention_config.compute_hash()
    bridge._configure_indexer_kv_dtype(cfg)
    assert cfg.attention_config.compute_hash() != original_hash
    assert cfg.attention_config.compute_hash() == core.AttentionConfig(
        indexer_kv_dtype="mxfp4").compute_hash()
    serializer = TypeAdapter(core.AttentionConfig)
    restored = serializer.validate_json(serializer.dump_json(cfg.attention_config))
    assert restored.indexer_kv_dtype == "mxfp4"


def configure_child(queue, cfg):
    # Forked reports must be reactivated locally before choosing a new default.
    bridge._RUNTIME_ACTIONS = bridge.ProcessLocalActions()
    publish(reports())
    bridge._configure_indexer_kv_dtype(cfg)
    queue.put(cfg.attention_config.indexer_kv_dtype)


@pytest.mark.parametrize("method", ["spawn", "fork"])
@pytest.mark.parametrize("dtype,expected", [
    ("auto", "mxfp4"), ("fp8", "fp8"), ("mxfp4", "mxfp4"),
])
def test_child_config_selection(method, dtype, expected):
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(method + " unavailable on this OS")
    context = multiprocessing.get_context(method)
    queue = context.Queue()
    process = context.Process(target=configure_child, args=(queue, config(dtype)))
    try:
        process.start()
        assert queue.get(timeout=30) == expected
        process.join(timeout=30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()
