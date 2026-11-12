# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools import l15_manifest as manifests
from tools.l15_ci_plan import ROOT, CURRENT, plan
from tools.l15_family_matrix import load


@pytest.mark.parametrize("families", [{"dense"}, {"mla", "mamba"},
                                      {"dense", "mla", "moe", "quantization", "mamba"}])
def test_ownership_is_disjoint_and_covers_every_original_green_node(families):
    baseline = manifests.load_manifest(ROOT / "tests/manifests/l15_v0522_classified.json")
    matrix = load(ROOT / "tests/manifests/l15_family_matrix_v1.json")
    original = manifests._expanded_tests(baseline)
    derived, ownership = plan(baseline, matrix, families)
    expected = {t["nodeid"] for t in original if t.get("source_id") in CURRENT
                and manifests._classification(t, baseline["defaults"])
                in {"core-green", "contract-green"}}
    exact, transferred = set(ownership["exact_gate"]), set(ownership["family_gate"])
    assert not exact & transferred
    assert exact | transferred == expected
    assert {t["nodeid"] for t in derived["tests"]} == exact
    observations = ownership["baseline_observations_not_executed"]
    assert all(manifests._classification(t, baseline["defaults"])
               not in {"core-green", "contract-green"} for t in observations)
    observed = {t["nodeid"] for t in ownership["observation_gate"]}
    not_executed = {t["nodeid"] for t in observations}
    assert not (exact | transferred) & (observed | not_executed)
    assert not observed & not_executed
    assert exact | transferred | observed | not_executed == {
        t["nodeid"] for t in original if t.get("source_id") in CURRENT}
    assert manifests._expanded_tests(baseline) == original
    assert not manifests.validate_manifest(derived)


def test_unknown_family_or_missing_source_is_refused():
    baseline = manifests.load_manifest(ROOT / "tests/manifests/l15_v0522_classified.json")
    matrix = load(ROOT / "tests/manifests/l15_family_matrix_v1.json")
    with pytest.raises(ValueError, match="families"):
        plan(baseline, matrix, {"typo"})
    baseline["test_groups"] = [g for g in baseline["test_groups"]
                               if g["source_id"] != "mechanism-current"]
    with pytest.raises(ValueError, match="source groups"):
        plan(baseline, matrix, {"dense"})
