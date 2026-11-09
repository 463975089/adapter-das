# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-length coverage and probability-policy negative controls."""

import copy

import pytest
import torch

from tools.checkpoint_accuracy import (
    POLICY,
    REFERENCE_MASS_TOLERANCE,
    cases,
    compare_top20,
    validate_cases,
    verdict,
)


class Tokenizer:
    chat_template = "test-template"

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs):
        ids = self.encode(messages[0]["content"])
        # Transformers 5.12 returns a mapping by default. The caller must
        # request the token-list protocol explicitly.
        return {"input_ids": ids} if kwargs.get("return_dict", True) else ids


@pytest.mark.parametrize("chat", [True, False])
def test_cases_include_exact_lengths_reuse_and_independent_changed_request(chat):
    tokenizer = Tokenizer()
    tokenizer.chat_template = "test-template" if chat else None
    batches = cases(tokenizer)
    validate_cases(batches)
    assert [len(c["input_ids"]) for c in batches[1]] == [512, 2048]
    assert batches[1][0]["input_ids"] == batches[2][0]["input_ids"]
    assert batches[1][1]["input_ids"] != batches[2][1]["input_ids"]
    changed = copy.deepcopy(batches)
    changed[2][0]["input_ids"][0] += 1
    with pytest.raises(ValueError, match="reused"):
        validate_cases(changed)
    changed = copy.deepcopy(batches)
    changed[1][1]["input_ids"].pop()
    with pytest.raises(ValueError, match="length"):
        validate_cases(changed)


def _head(logprobs, ids):
    return [[[logprobs[0, token].item(), token, None] for token in ids]]


def test_coarsened_distribution_is_rank_free_and_exposes_probability_corruption():
    lp = torch.linspace(3, -3, 32)[None].log_softmax(-1)
    observed = _head(lp, list(reversed(range(20))))
    good = compare_top20(lp, observed)
    assert all(value <= 1e-12 for value in good.values())
    # Large-vocabulary FP32 log_softmax can accumulate small re-sum drift.
    drifted = lp + (REFERENCE_MASS_TOLERANCE / 2)
    compare_top20(drifted, observed)
    bad = copy.deepcopy(observed)
    bad[0][0][0], bad[0][-1][0] = bad[0][-1][0], bad[0][0][0]
    broken = compare_top20(lp, bad)
    assert broken["max_top20_logprob_error"] > .15
    assert broken["max_top20_tail_jsd"] > .002
    # True probabilities for 20 arbitrary tail labels alone would produce
    # zero divergence; the missed head-mass condition catches that omission.
    omitted = compare_top20(lp, _head(lp, list(range(12, 32))))
    assert omitted["max_top20_logprob_error"] == 0
    assert omitted["max_top20_missed_mass"] > .02


@pytest.mark.parametrize(
    "fault",
    ["missing", "duplicate", "nan", "positive", "mass", "vocab", "reference_mass"],
)
def test_invalid_top20_is_refused(fault):
    lp = torch.linspace(3, -3, 32)[None].log_softmax(-1)
    observed = _head(lp, list(range(20)))
    if fault == "missing":
        observed[0].pop()
    elif fault == "duplicate":
        observed[0][0][1] = observed[0][1][1]
    elif fault == "nan":
        observed[0][0][0] = float("nan")
    elif fault == "positive":
        observed[0][0][0] = .1
    elif fault == "mass":
        for row in observed[0]:
            row[0] = -.01
    elif fault == "reference_mass":
        lp = lp + (REFERENCE_MASS_TOLERANCE * 2)
    else:
        observed[0][0][1] = 32
    with pytest.raises(ValueError):
        compare_top20(lp, observed)


def test_policy_rejects_outliers_and_nan_even_after_valid_rows():
    row = {key: 0. for key in POLICY}
    assert verdict([row])["passed"]
    outlier = dict(row, max_greedy_regret=.2)
    assert not verdict([row, outlier])["passed"]
    for value in (float("nan"), float("inf"), -.01):
        with pytest.raises(ValueError, match="finite"):
            verdict([row, dict(row, max_top20_tail_jsd=value)])
