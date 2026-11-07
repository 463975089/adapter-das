"""Frozen-prefix comparisons must not compare different token histories."""

import copy
import json

import pytest

from tools.l15_decode_consistency import compare, main


def record(prompt=(1, 2), output=(3, 4, 5)):
    heads = [[[-0.1, token, None], [-3., 8, None]] for token in output]
    return {
        "input_ids": list(prompt),
        "result": {
            "output_ids": list(output),
            "meta_info": {
                "prompt_tokens": len(prompt), "completion_tokens": len(output),
                "cached_tokens": 0,
                "output_token_logprobs": [head[0][:] for head in heads],
                "output_top_logprobs": heads,
            },
        },
    }


def test_same_ids_can_hide_probability_drift():
    left, right = record(), record()
    right["result"]["meta_info"]["output_top_logprobs"][1][1][0] = -2.
    report = compare(left, right)
    assert report["aligned_count"] == 3
    assert report["first_observed_difference"] == {
        "left_output_position": 1, "right_output_position": 1}
    assert report["positions"][1]["selected_logprob_delta"] == 0
    assert report["positions"][1]["max_shared_logprob_delta"] == 1
    assert report["model_acceptance"] is False
    assert "passed" not in report


def test_divergence_compares_its_prediction_but_not_later_histories():
    report = compare(record(), record(output=(3, 6, 5)))
    assert report["aligned_count"] == 2
    assert report["left_unaligned_count"] == report["right_unaligned_count"] == 1
    assert report["positions"][-1]["selected_ids_equal"] is False
    assert report["positions"][-1]["selected_logprob_delta"] is None


@pytest.mark.parametrize("position", range(3))
def test_frozen_prefix_prefill_aligns_correct_incremental_position(position):
    incremental = record()
    teacher = record(prompt=(1, 2, 3, 4)[:2 + position],
                     output=((3, 4, 5)[position],))
    report = compare(incremental, teacher)
    assert report["aligned_count"] == 1
    assert report["positions"][0]["left_output_position"] == position
    assert report["positions"][0]["right_output_position"] == 0
    assert report["first_observed_difference"] is None


def test_reconverged_output_ids_do_not_realign_different_histories():
    report = compare(record(output=(3, 4, 5)), record(output=(6, 4, 5)))
    assert report["aligned_count"] == 1


def test_disjoint_top_sets_are_missing_evidence_not_zero_error():
    # Remove the only shared top token to exercise an empty intersection.
    other = record(output=(6,))
    other["result"]["meta_info"]["output_top_logprobs"][0][1][1] = 9
    report = compare(record(output=(3,)), other)
    assert report["positions"][0]["shared_reported_count"] == 0
    assert report["positions"][0]["max_shared_logprob_delta"] is None
    assert report["first_observed_difference"] is not None


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(input_ids=[True, 2]),
    lambda r: r["result"].update(output_ids=[-1]),
    lambda r: r["result"]["meta_info"].update(prompt_tokens=True),
    lambda r: r["result"]["meta_info"].update(completion_tokens=4),
    lambda r: r["result"]["meta_info"].update(cached_tokens=None),
    lambda r: r["result"]["meta_info"].update(cached_tokens=3),
    lambda r: r["result"]["meta_info"].update(output_token_logprobs=[]),
    lambda r: r["result"]["meta_info"].update(output_top_logprobs=[[]] * 3),
    lambda r: r["result"]["meta_info"]["output_token_logprobs"][0].__setitem__(1, 7),
    lambda r: r["result"]["meta_info"]["output_token_logprobs"][0].__setitem__(0, float("nan")),
    lambda r: r["result"]["meta_info"]["output_top_logprobs"][0][1].__setitem__(0, float("inf")),
    lambda r: r["result"]["meta_info"]["output_top_logprobs"][0][1].__setitem__(0, .1),
    lambda r: r["result"]["meta_info"]["output_top_logprobs"][0][1].__setitem__(1, 3),
    lambda r: r["result"]["meta_info"]["output_top_logprobs"][0][0].__setitem__(0, -.2),
])
def test_malformed_probabilities_or_histories_rejected(mutation):
    invalid = copy.deepcopy(record())
    mutation(invalid)
    with pytest.raises(ValueError):
        compare(record(), invalid)


def test_teacher_with_different_prefix_is_rejected():
    with pytest.raises(ValueError, match="no identical"):
        compare(record(), record(prompt=(1, 2, 9), output=(4,)))


def test_cli_binds_input_bytes_and_preserves_existing_evidence(tmp_path):
    left, right, out = (tmp_path / name for name in ("left.json", "right.json", "out.json"))
    for path in (left, right):
        path.write_text(json.dumps(record()), encoding="utf-8")
    argv = ["--left", str(left), "--right", str(right), "--out", str(out)]
    assert main(argv) == 0
    report = json.loads(out.read_text())
    assert len(report["input_sha256"]["left"]) == 64
    assert report["input_sha256"]["left"] == report["input_sha256"]["right"]
    original = out.read_bytes()
    with pytest.raises(FileExistsError):
        main(argv)
    assert out.read_bytes() == original
    with pytest.raises(FileExistsError):
        main(argv[:-1] + [str(left)])
