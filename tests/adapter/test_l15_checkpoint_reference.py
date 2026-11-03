"""Reference probes must reject weak evidence and expose probability errors."""

import copy
from types import SimpleNamespace

import pytest
import torch

from tools.l15_checkpoint_reference import compare, fingerprint, requests, run


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
