"""Reference probes must reject weak evidence and expose probability errors."""

import copy
from types import SimpleNamespace

import pytest
import torch

from tools.l15_checkpoint_reference import compare, execution_binding, fingerprint, requests, run


def _report():
    jobs = []
    for mode in ("eager", "graph"):
        batch = [
            {"output_ids": [1] * 8, "meta_info": {
                "output_token_logprobs": [[-.1, 1, None]] * 8,
                "input_token_logprobs": [[None, 2, None], [-.2, 3, None]],
                "prompt_tokens": 2, "cached_tokens": 0,
            }} for _ in range(2)
        ]
        jobs.append({
            "mode": mode, "passed": True, "returncode": 0, "timed_out": False,
            "evidence": {
                "artifact": "l15-model-worker", "passed": True, "mode": mode,
                "family": "checkpoint", "weight_evidence": "real-checkpoint",
                "model": "/model/dense",
                "generations": [batch, copy.deepcopy(batch)],
                "server_info": {"internal_states": [{
                    "decode_cuda_graph_bs": [],
                    "memory_usage": {"graph": {"decode": .02 if mode == "graph" else 0}},
                }]},
            },
        })
    return {
        "artifact": "l15-model-smoke", "passed": True, "complete": True,
        "families": ["checkpoint"], "modes": ["eager", "graph"], "jobs": jobs,
        "runtime": {"source_integrity": {"passed": True}},
    }


def test_real_eager_graph_requests_keep_exact_prompt_and_output_ids():
    _, rows = requests(_report())
    assert len(rows) == 8 and sum(len(r["output_ids"]) for r in rows) == 64
    assert {r["mode"] for r in rows} == {"eager", "graph"}
    assert all(r["prompt_ids"] == [2, 3] and r["output_ids"] == [1] * 8 for r in rows)


@pytest.mark.parametrize("fault", [
    "dummy", "source", "duplicate", "duplicate-modes", "wave", "cached", "prompt", "nan", "failed",
])
def test_weak_or_incomplete_smoke_is_refused(fault):
    report = _report()
    evidence = report["jobs"][0]["evidence"]
    meta = evidence["generations"][0][0]["meta_info"]
    if fault == "dummy":
        evidence["weight_evidence"] = "dummy"
    elif fault == "source":
        report["runtime"]["source_integrity"]["passed"] = False
    elif fault == "duplicate":
        report["jobs"][1] = copy.deepcopy(report["jobs"][0])
    elif fault == "duplicate-modes":
        report["modes"].append("graph")
    elif fault == "wave":
        evidence["generations"].pop()
    elif fault == "cached":
        meta["cached_tokens"] = 1
    elif fault == "prompt":
        meta["input_token_logprobs"].pop()
    elif fault == "nan":
        meta["output_token_logprobs"][0][0] = float("nan")
    else:
        report["jobs"][0]["returncode"] = 1
    with pytest.raises(ValueError):
        requests(report)


def test_reference_metrics_detect_wrong_scores_and_non_greedy_tokens():
    lp = torch.tensor([[5., -4., 0.], [-4., 5., 0.]]).log_softmax(-1)
    observed = [lp[0, 0].item(), lp[1, 1].item()]
    same = compare(lp, [0, 1], observed)
    assert same["max_logprob_error"] == 0 and same["max_greedy_regret"] == 0
    assert same["reference_top1_matches"] == 2
    wrong_score = compare(lp, [0, 1], [v - 1 for v in observed])
    assert wrong_score["max_logprob_error"] == pytest.approx(1.)
    # Even matching the probability of a wrong sampled token is not greedy.
    wrong_token = compare(lp, [1, 0], [lp[0, 1].item(), lp[1, 0].item()])
    assert wrong_token["max_logprob_error"] == 0
    assert wrong_token["max_greedy_regret"] == pytest.approx(9.)
    assert wrong_token["reference_top1_matches"] == 0
    with pytest.raises(ValueError, match="invalid reference"):
        compare(lp * float("nan"), [0, 1], observed)


def test_reference_fingerprint_detects_weight_and_config_changes(tmp_path):
    with pytest.raises(ValueError, match="real safetensors"):
        fingerprint(tmp_path)
    (tmp_path / "config.json").write_text('{"model_type":"qwen2"}')
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"original reference weight bytes")
    first = fingerprint(tmp_path)
    shard.write_bytes(b"changed reference weight bytes")
    assert fingerprint(tmp_path)["model.safetensors"] != first["model.safetensors"]
    assert fingerprint(tmp_path)["config.json"] == first["config.json"]
    (tmp_path / "config.json").write_text('{"model_type":"qwen3"}')
    assert fingerprint(tmp_path)["config.json"] != first["config.json"]


def test_reference_output_cannot_replace_original_smoke_evidence(tmp_path):
    source = tmp_path / "smoke.json"
    source.write_text("original evidence")
    with pytest.raises(ValueError, match="preserve the original"):
        run(SimpleNamespace(out=source, smoke_report=source))
    assert source.read_text() == "original evidence"


def test_reference_binding_requires_controller_and_worker_agreement():
    report = _report()
    model, _ = requests(report)
    assert execution_binding(report, model) is None
    bound = {"config.json": "abc", "model.safetensors": "def"}
    report["runtime"]["checkpoints"] = {str(model): bound}
    with pytest.raises(ValueError, match="inconsistent HCU"):
        execution_binding(report, model)
    for job in report["jobs"]:
        job["evidence"]["checkpoint_files_sha256"] = dict(bound)
        job["checkpoint_integrity_after"] = {"passed": True, "files_sha256": dict(bound)}
    assert execution_binding(report, model) == bound
    report["jobs"][1]["evidence"]["checkpoint_files_sha256"]["model.safetensors"] = "changed"
    with pytest.raises(ValueError, match="inconsistent HCU"):
        execution_binding(report, model)


def test_reference_cannot_mix_output_precision_configurations():
    report = _report()
    report["enable_fp32_lm_head"] = True
    with pytest.raises(ValueError, match="worker"):
        requests(report)
    for job in report["jobs"]:
        job["evidence"]["enable_fp32_lm_head"] = True
    with pytest.raises(ValueError, match="LM-head"):
        requests(report)
    for job in report["jobs"]:
        job["evidence"]["server_info"]["enable_fp32_lm_head"] = True
    assert len(requests(report)[1]) == 8


def test_reference_cannot_mix_model_dtypes_or_reuse_legacy_bf16_as_fp32():
    report = _report()
    report["dtype"] = "float32"
    with pytest.raises(ValueError, match="worker"):
        requests(report)
    for job in report["jobs"]:
        job["evidence"]["dtype"] = "float32"
    with pytest.raises(ValueError, match="runtime dtype"):
        requests(report)
    for job in report["jobs"]:
        job["evidence"]["server_info"]["dtype"] = "float32"
    assert len(requests(report)[1]) == 8


def test_reference_cannot_mix_explicit_and_automatic_quantization_evidence():
    report = _report()
    report["quantization"] = "w8a8_int8"
    with pytest.raises(ValueError, match="worker"):
        requests(report)
    for job in report["jobs"]:
        job["evidence"]["quantization"] = "w8a8_int8"
    with pytest.raises(ValueError, match="runtime quantization"):
        requests(report)
    for job in report["jobs"]:
        job["evidence"]["server_info"]["quantization"] = "w8a8_int8"
    assert len(requests(report)[1]) == 8


def test_explicit_online_quantization_cannot_bypass_unquantized_reference_guard(tmp_path, monkeypatch, request):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    from tools import l15_checkpoint_reference as reference
    from hcu_adapter import hcu_env

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"vocab_size":32}')
    source = tmp_path / "smoke.json"
    source.write_text('{"quantization":"w8a8_int8"}')
    monkeypatch.setattr(hcu_env, "resolve", lambda _: ("none", "test"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(reference, "requests", lambda _: (model, []))
    monkeypatch.setattr(reference, "execution_binding", lambda *_: None)
    with pytest.raises(ValueError, match="quantized checkpoints or execution"):
        run(SimpleNamespace(smoke_report=source, out=tmp_path / "out.json", threads=1))


@pytest.mark.parametrize("fault", ["execution-override", "unbound"])
def test_ct_reference_requires_matching_execution_and_byte_binding(
    tmp_path, monkeypatch, request, fault,
):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    import json
    from test_checkpoint_quantization_reference import configuration
    from tools import l15_checkpoint_reference as reference
    from hcu_adapter import hcu_env

    model = tmp_path / "model"
    model.mkdir()
    config = configuration()
    config["vocab_size"] = 32
    (model / "config.json").write_text(json.dumps(config))
    source = tmp_path / "smoke.json"
    source.write_text(json.dumps({
        "quantization": "w8a8_int8" if fault == "execution-override" else None,
    }))
    monkeypatch.setattr(hcu_env, "resolve", lambda _: ("none", "test"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(reference, "requests", lambda _: (model, []))
    monkeypatch.setattr(reference, "execution_binding",
                        lambda *_: None if fault == "unbound" else {"config.json": "test"})
    with pytest.raises(ValueError, match="does not match|bound to HCU execution"):
        run(SimpleNamespace(smoke_report=source, out=tmp_path / "out.json",
                            threads=1, quantized_reference="ct-fp8-token-channel"))


def test_accuracy_requests_use_explicit_ids_for_cached_prompts_and_require_top20():
    from tools.checkpoint_accuracy import cases

    class Tokenizer:
        chat_template = None

        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

    report = _report()
    report["generation_profile"] = "accuracy"
    vectors = cases(Tokenizer())
    for job in report["jobs"]:
        job["source_integrity_after"] = {"passed": True}
        evidence = job["evidence"]
        evidence.update(generation_profile="accuracy", accuracy_cases=vectors)
        original = evidence["generations"][0]
        evidence["generations"] = [copy.deepcopy(original) for _ in range(3)]
        for wave, batch in enumerate(evidence["generations"]):
            for request, item in enumerate(batch):
                meta = item["meta_info"]
                meta["prompt_tokens"] = len(vectors[wave][request]["input_ids"])
                meta["cached_tokens"] = 500 if wave == 2 else 0
                meta["input_token_logprobs"] = []
                meta["output_top_logprobs"] = [
                    [[-.1, 1, None]] + [[-10., token, None] for token in range(2, 21)]
                    for _ in range(8)]
    _, rows = requests(report)
    assert len(rows) == 12
    assert rows[2]["prompt_ids"] == rows[4]["prompt_ids"]
    assert [len(r["prompt_ids"]) for r in rows[2:4]] == [512, 2048]
    report["jobs"][0]["evidence"]["generations"][0][0]["meta_info"]["output_top_logprobs"] = []
    with pytest.raises(ValueError, match="missing generated"):
        requests(report)
