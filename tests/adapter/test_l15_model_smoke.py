# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Offline failure contracts for the real-model smoke launcher."""

import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.l15_model_smoke import tiny_config, validate_generation, validate_lm_head_option, validate_mode
from tools.l15_runtime import run_job
from tools import l15_model_smoke as smoke
from tools.checkpoint_files import audit, fingerprint


def test_tiny_models_are_separate_architectures():
    configs = [tiny_config(f) for f in ("dense", "moe", "mla", "mamba", "hybrid")]
    assert len({c["architectures"][0] for c in configs}) == 5
    assert all(c["num_hidden_layers"] == 2 for c in configs)
    # Upstream fused RoPE's admitted head sizes begin at 64.
    assert configs[1]["head_dim"] == 64
    assert configs[-1]["layer_types"] == ["linear_attention", "full_attention"]
    with pytest.raises(ValueError, match="unknown"):
        tiny_config("typo")


def test_default_binding_manifest_is_anchored_to_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = smoke.build_parser().parse_args(["--up-tree", str(tmp_path)])
    assert args.binding_manifest == smoke.DEFAULT_BINDING_MANIFEST
    assert args.binding_manifest.is_file()


@pytest.mark.parametrize("fault", ["batch", "tokens", "vocab", "logprobs", "nan", "prob-token"])
def test_invalid_generation_is_never_a_pass(fault):
    document = [
        {"output_ids": [7] * 8,
         "meta_info": {"output_token_logprobs": [[-1.0, 7, None]] * 8}}
        for _ in range(2)
    ]
    validate_generation(document, 256)
    broken = copy.deepcopy(document)
    if fault == "batch":
        broken.pop()
    elif fault == "tokens":
        broken[0]["output_ids"].pop()
    elif fault == "vocab":
        broken[0]["output_ids"][0] = 256
    elif fault == "logprobs":
        broken[0]["meta_info"]["output_token_logprobs"] = []
    elif fault == "nan":
        broken[0]["meta_info"]["output_token_logprobs"][0][0] = float("nan")
    else:
        broken[0]["meta_info"]["output_token_logprobs"][0][1] = 8
    with pytest.raises(ValueError):
        validate_generation(broken, 256)


def test_graph_flag_without_capture_is_not_graph_evidence():
    validate_mode({"internal_states": [{"decode_cuda_graph_bs": [],
                   "memory_usage": {"graph": {"decode": 0.02}}}]}, "graph")
    validate_mode({"internal_states": [{"decode_cuda_graph_bs": [],
                   "memory_usage": {"graph": {"decode": 0}}}]}, "eager")
    for info in ({}, {"internal_states": [{}]},
                 {"internal_states": [{"decode_cuda_graph_bs": []}]},
                 {"internal_states": [{"decode_cuda_graph_bs": [1, 2],
                                      "memory_usage": {"graph": {"decode": 0}}}]}):
        with pytest.raises(ValueError):
            validate_mode(info, "graph")


def test_output_precision_must_match_actual_server_configuration():
    validate_lm_head_option({}, False)  # legacy default evidence
    validate_lm_head_option({"enable_fp32_lm_head": True}, True)
    with pytest.raises(ValueError, match="LM-head"):
        validate_lm_head_option({"enable_fp32_lm_head": False}, True)
    with pytest.raises(ValueError, match="LM-head"):
        validate_lm_head_option({"enable_fp32_lm_head": True}, False)


def test_execution_dtype_must_match_actual_server_and_reject_unknown_precision():
    smoke.validate_execution_dtype({}, "bfloat16")
    smoke.validate_execution_dtype({"dtype": "float32"}, "float32")
    for info, dtype in [({}, "float32"), ({"dtype": "float32"}, "bfloat16"),
                        ({"dtype": "bfloat16"}, "float32"), ({}, "float16")]:
        with pytest.raises(ValueError, match="runtime dtype"):
            smoke.validate_execution_dtype(info, dtype)


@pytest.mark.parametrize("value", [0, 1, -.1, 1.1, float("nan"), float("inf"),
                                  True, False, "0.7", None])
def test_memory_fraction_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="memory fraction"):
        smoke.validate_memory_fraction(value)


def test_memory_fraction_must_match_actual_server():
    smoke.validate_memory_option({}, .25)  # Only the historical default is implied.
    smoke.validate_memory_option({"mem_fraction_static": .7}, .7)
    for info, value in [({}, .7), ({"mem_fraction_static": .25}, .7),
                        ({"mem_fraction_static": .7}, .25),
                        ({"mem_fraction_static": True}, .7)]:
        with pytest.raises(ValueError, match="memory fraction"):
            smoke.validate_memory_option(info, value)


def test_explicit_quantization_requires_actual_server_agreement():
    smoke.validate_quantization_option({}, None)
    smoke.validate_quantization_option({"quantization": "w8a8_int8"}, "w8a8_int8")
    for info, requested in [({}, "w8a8_int8"), ({"quantization": "fp8"}, "w8a8_int8")]:
        with pytest.raises(ValueError, match="runtime quantization"):
            smoke.validate_quantization_option(info, requested)


def test_tp_size_requires_actual_server_and_scheduler_agreement():
    smoke.validate_tp_option({}, 1)  # historical single-device evidence
    smoke.validate_tp_option(
        {"tp_size": 2, "internal_states": [{"tp_size": 2}]}, 2)
    for value in (True, False, 0, 3, -1, "2"):
        with pytest.raises(ValueError, match="TP size"):
            smoke.validate_tp_size(value)
    for info in ({}, {"tp_size": 1}, {"tp_size": True},
                 {"tp_size": 2, "internal_states": [{}]},
                 {"tp_size": 2, "internal_states": [{"tp_size": 1}]}):
        with pytest.raises(ValueError, match="TP size"):
            smoke.validate_tp_option(info, 2)


def _checkpoint(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"vocab_size":256}')
    (model / "model.safetensors").write_bytes(b"checkpoint weights")
    return model


def test_incomplete_and_escaping_shard_indices_are_refused(tmp_path):
    model = _checkpoint(tmp_path)
    index = model / "model.safetensors.index.json"
    for name in ("missing.safetensors", "../escape.safetensors", "/absolute.safetensors"):
        index.write_text(json.dumps({"weight_map": {"weight": name}}))
        with pytest.raises(ValueError, match="shard"):
            fingerprint(model)
    index.write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    expected = fingerprint(model)
    (model / "model.safetensors").unlink()
    failed = audit(model, expected)
    assert failed["passed"] is False and "missing checkpoint shard" in failed["error"]


@pytest.mark.parametrize("coarse_metadata", [False, True])
def test_hashing_rechecks_assets_already_read(tmp_path, monkeypatch, coarse_metadata):
    import tools.checkpoint_files as files

    model = _checkpoint(tmp_path)
    original = files.hashlib.sha256
    if coarse_metadata:
        monkeypatch.setattr(files, "_identity", lambda stat: "same-timestamp-tick")

    class Digest:
        def __init__(self):
            self.impl = original()

        def update(self, data):
            if data == b"checkpoint weights":
                # config.json was hashed first; mutate it while hashing weights.
                (model / "config.json").write_text('{"vocab_size":512}')
            self.impl.update(data)

        def hexdigest(self):
            return self.impl.hexdigest()

    monkeypatch.setattr(files.hashlib, "sha256", Digest)
    with pytest.raises(ValueError, match="changed while hashing other"):
        fingerprint(model)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("quantization", [None, "w8a8_int8"])
@pytest.mark.parametrize("change", ["clean", "before", "during"])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("mem_fraction", [.25, .7])
def test_worker_binds_bytes_before_loading_and_after_execution(
    tmp_path, monkeypatch, change, dtype, quantization, tp_size, mem_fraction
):
    model = _checkpoint(tmp_path)
    expected = fingerprint(model)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(expected))
    created = []

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs["dtype"] == dtype
            assert kwargs.get("quantization") == quantization
            assert kwargs["tp_size"] == tp_size
            assert kwargs["mem_fraction_static"] == mem_fraction
            created.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def generate(self, **kwargs):
            if change == "during":
                (model / "model.safetensors").write_bytes(b"changed during generation")
            return [{"output_ids": [7] * 8,
                     "meta_info": {"output_token_logprobs": [[-.1, 7, None]] * 8}}
                    for _ in range(2)]

        def get_server_info(self):
            return {"dtype": dtype, "quantization": quantization, "tp_size": tp_size,
                    "mem_fraction_static": mem_fraction,
                    "internal_states": [{"tp_size": tp_size, "decode_cuda_graph_bs": [],
                    "memory_usage": {"graph": {"decode": 0}}}]}

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=Engine))
    args = SimpleNamespace(model=model, family="checkpoint", mode="eager",
                           checkpoint_manifest=manifest, out=tmp_path / "worker.json",
                           dtype=dtype, quantization=quantization, tp_size=tp_size,
                           mem_fraction_static=mem_fraction)
    if change == "before":
        (model / "model.safetensors").write_bytes(b"changed before loading")
    if change == "clean":
        smoke.worker(args)
        evidence = json.loads(args.out.read_text())
        assert evidence["passed"] and evidence["checkpoint_files_sha256"] == expected
        assert evidence["dtype"] == dtype
        assert evidence["quantization"] == quantization
        assert evidence["tp_size"] == tp_size
        assert evidence["mem_fraction_static"] == mem_fraction
    else:
        with pytest.raises(ValueError, match="checkpoint changed"):
            smoke.worker(args)
        assert not args.out.exists()
        assert bool(created) == (change == "during")


@pytest.mark.parametrize("fp32_head", [False, True])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("quantization", [None, "w8a8_int8"])
@pytest.mark.parametrize("mem_fraction", [.25, .7])
@pytest.mark.parametrize("fault", [
    "clean", "worker-unbound", "after", "head-ignored", "dtype-ignored", "quantization-ignored",
    "memory-ignored",
])
def test_controller_cannot_accept_unbound_or_changed_checkpoint(
    tmp_path, monkeypatch, fault, fp32_head, dtype, quantization, mem_fraction
):
    model = _checkpoint(tmp_path)
    monkeypatch.setattr(smoke, "preflight",
                        lambda *args, **kwargs: {"upstream": {"commit": "community"}})
    monkeypatch.setattr(smoke, "audit_source", lambda *args: {"passed": True})

    def job(argv, **kwargs):
        assert argv[argv.index("--tp-size") + 1] == "1"
        assert float(argv[argv.index("--mem-fraction-static") + 1]) == mem_fraction
        assert ("--enable-fp32-lm-head" in argv) is fp32_head
        assert argv[argv.index("--dtype") + 1] == dtype
        assert ("--quantization" in argv) == (quantization is not None)
        if quantization is not None:
            assert argv[argv.index("--quantization") + 1] == quantization
        manifest = Path(argv[argv.index("--checkpoint-manifest") + 1])
        evidence = {
            "artifact": "l15-model-worker", "passed": True, "family": "checkpoint",
            "mode": argv[argv.index("--mode") + 1], "model": str(model),
            "weight_evidence": "real-checkpoint",
            "enable_fp32_lm_head": fp32_head if fault != "head-ignored" else False,
            "dtype": dtype if fault != "dtype-ignored" else "bfloat16",
            "quantization": quantization if fault != "quantization-ignored" else None,
            "mem_fraction_static": mem_fraction if fault != "memory-ignored" else .25,
        }
        if fault != "worker-unbound":
            evidence["checkpoint_files_sha256"] = json.loads(manifest.read_text())
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(evidence))
        if fault == "after":
            (model / "model.safetensors").write_bytes(b"changed after worker")
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(smoke, "run_job", job)
    args = SimpleNamespace(
        up_tree=tmp_path, checkpoint=model, families="checkpoint", modes="eager,graph",
        binding_manifest=tmp_path / "bindings.json", out=tmp_path / "report.json", timeout=10,
        enable_fp32_lm_head=fp32_head,
        dtype=dtype,
        quantization=quantization,
        mem_fraction_static=mem_fraction,
    )
    expected = (fault == "clean" or (fault == "head-ignored" and not fp32_head)
                or (fault == "dtype-ignored" and dtype == "bfloat16")
                or (fault == "quantization-ignored" and quantization is None)
                or (fault == "memory-ignored" and mem_fraction == .25))
    assert smoke.run(args) == expected
    report = json.loads(args.out.read_text())
    assert report["passed"] == expected
    assert report["enable_fp32_lm_head"] is fp32_head
    assert report["dtype"] == dtype
    assert report["quantization"] == quantization
    assert report["mem_fraction_static"] == mem_fraction
    if fault == "after":
        assert not report["complete"] and len(report["jobs"]) == 1
        assert report["jobs"][0]["checkpoint_integrity_after"]["passed"] is False


@pytest.mark.parametrize("fault", ["clean", "ignored", "boolean", "missing"])
@pytest.mark.parametrize("host", ["adapter", "adapter+plugin"])
def test_tp2_controller_binds_worker_parallelism(tmp_path, monkeypatch, fault, host):
    model = _checkpoint(tmp_path)

    def preflight(*args, **kwargs):
        assert kwargs == {"host": host, "device_count": 2}
        return {"upstream": {"commit": "community"}, "device_count": 2}

    def job(argv, **kwargs):
        assert argv[argv.index("--tp-size") + 1] == "2"
        assert argv[argv.index("--host") + 1] == host
        assert kwargs["env"]["SGLANG_PLUGINS"] == (
            "hcu" if host == "adapter+plugin" else "none")
        manifest = Path(argv[argv.index("--checkpoint-manifest") + 1])
        evidence = {
            "artifact": "l15-model-worker", "passed": True, "family": "checkpoint",
            "mode": argv[argv.index("--mode") + 1], "model": str(model),
            "weight_evidence": "real-checkpoint",
            "host": host,
            "checkpoint_files_sha256": json.loads(manifest.read_text()),
        }
        if fault != "missing":
            evidence["tp_size"] = {"clean": 2, "ignored": 1, "boolean": True}[fault]
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(evidence))
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(smoke, "preflight", preflight)
    monkeypatch.setattr(smoke, "audit_source", lambda *args: {"passed": True})
    monkeypatch.setattr(smoke, "run_job", job)
    args = SimpleNamespace(
        up_tree=tmp_path, checkpoint=model, families="checkpoint", modes="eager,graph",
        binding_manifest=tmp_path / "bindings.json", out=tmp_path / "report.json",
        timeout=10, tp_size=2, host=host,
    )
    assert smoke.run(args) is (fault == "clean")
    report = json.loads(args.out.read_text())
    assert report["tp_size"] == 2
    assert report["passed"] is (fault == "clean")


@pytest.mark.skipif(os.name != "posix", reason="remote Linux job lifecycle")
@pytest.mark.parametrize("nested", [False, True])
def test_timeout_stops_own_descendants(tmp_path, nested):
    script = (
        "import subprocess,sys,time\n"
        "p=subprocess.Popen([sys.executable,'-S','-c','import time;time.sleep(120)'])\n"
        "print(p.pid,flush=True)\n"
        "time.sleep(120)\n"
    )
    log = tmp_path / "job.log"
    pid_log = log
    env = dict(os.environ)
    if nested:
        pid_log = tmp_path / "inner.log"
        script = (
            "import os,sys\n"
            "from tools.l15_runtime import run_job\n"
            f"run_job([sys.executable,'-S','-c',{script!r}],cwd={str(tmp_path)!r},"
            f"env=dict(os.environ),timeout=100,log={str(pid_log)!r})\n"
        )
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = run_job(
        [sys.executable, "-S", "-c", script], cwd=tmp_path, env=env,
        timeout=2, log=log,
    )
    assert result["timed_out"] and result["returncode"] == 124
    child = int(pid_log.read_text().strip())
    stat = Path(f"/proc/{child}/stat")
    if stat.exists():
        assert stat.read_text().split()[2] == "Z", "timed-out child still executing"


@pytest.mark.parametrize("fault", ["clean", "tokens", "batches"])
def test_request_replay_rejects_state_leakage(fault):
    batches = [[{"output_ids": [7] * 8,
                 "meta_info": {"output_token_logprobs": [[-.1, 7, None]] * 8}}
                for _ in range(2)] for _ in range(3)]
    if fault == "tokens":
        batches[2][1]["output_ids"][0] = 8
    if fault == "batches":
        batches.pop()
    if fault == "clean":
        result = smoke.validate_request_replay(batches)
        assert result["passed"] and result["max_logprob_difference"] == 0
        assert result["cache_hit_proven"] is False
    else:
        with pytest.raises(ValueError, match="replay"):
            smoke.validate_request_replay(batches)


@pytest.mark.parametrize("fault", ["clean", "zero", "missing", "boolean", "tokens"])
def test_cache_probe_requires_actual_hit_and_stable_output(fault):
    class Engine:
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            assert kwargs["logprob_start_len"] == -1
            self.calls += 1
            count = {"clean": 3, "zero": 0, "missing": None,
                     "boolean": True, "tokens": 3}[fault]
            token = 8 if fault == "tokens" and self.calls == 2 else 7
            return [{"output_ids": [token] * 8,
                     "meta_info": {"cached_tokens": count,
                                   "output_token_logprobs": [[-.1, token, None]] * 8}}
                    for _ in range(2)]
    original = {"logprob_start_len": 0, "input_ids": [[1, 2], [3, 4]]}
    if fault == "clean":
        report = smoke.probe_cache_reuse(Engine(), original, 256)
        assert report["cache_hit_proven"] and report["cached_tokens"] == [3, 3]
    else:
        with pytest.raises(ValueError, match="cache"):
            smoke.probe_cache_reuse(Engine(), original, 256)
    assert original["logprob_start_len"] == 0
