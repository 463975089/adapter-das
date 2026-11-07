"""Original HIP compression regressions under the adapter's CUDA API facade."""

import ast
import importlib.util
import inspect
from pathlib import Path
import unittest

import pytest
import torch


CASES = {
    "pair-state": (
        "test_pair_state_survives_the_next_decode_step",
        "HIP fused pair-state writer",
    ),
    "padding": (
        "test_padded_rows_publish_nothing",
        "HIP fused compressor regression",
    ),
    "open-group": (
        "test_open_group_rows_publish_nothing",
        "HIP fused compressor regression",
    ),
}


def _run_required_case(case_class, method_name, reason, source):
    """Rehost one known method; all other skip/failure semantics remain strict."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == case_class.__name__)
    method = next(node for node in owner.body
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    assert len(method.decorator_list) == 1, "Upstream skip contract changed"
    decorator = method.decorator_list[0]
    expected = ast.parse(
        f"@unittest.skipUnless(is_hip(), {reason!r})\ndef check(self): pass"
    ).body[0].decorator_list[0]
    assert ast.dump(decorator) == ast.dump(expected), "Upstream skip contract changed"
    original = getattr(case_class, method_name)
    if getattr(original, "__unittest_skip__", False):
        assert original.__unittest_skip_why__ == reason
        body = original.__wrapped__
        assert not getattr(body, "__unittest_skip__", False)
    else:
        body = original
    assert body.__name__ == method_name
    assert tuple(inspect.signature(body).parameters) == ("self",)
    assert Path(body.__code__.co_filename).resolve() == source.resolve()
    # A subclass invokes the original body through unittest's real class/test
    # fixtures, subTest accounting and cleanup stack. The upstream class and its
    # decorated method are left intact.
    required = type("RequiredHcuCompressionCase", (case_class,), {method_name: body})
    result = unittest.TestResult()
    unittest.TestSuite([required(method_name)]).run(result)
    assert getattr(case_class, method_name) is original
    failures = "\n".join(text for _, text in result.errors + result.failures)
    assert (result.testsRun == 1 and not result.skipped and not result.expectedFailures
            and not result.unexpectedSuccesses and result.wasSuccessful()), (
        f"Original compression regression incomplete or failed: "
        f"runs={result.testsRun}, skipped={result.skipped}\n{failures}"
    )


@pytest.mark.parametrize("case_id", list(CASES))
@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="Requires a real HIP device",
)
def test_original_compression_state(case_id):
    import sglang

    # Resolve the test from the community package actually imported by this
    # process. The strict matrix binds and audits this source tree separately.
    root = Path(sglang.__file__).resolve().parents[2]
    source = root / "test/registered/attention/unittests/dsv4/test_dsv41_fused_compress.py"
    assert source.is_file(), f"Missing original community compression test: {source}"
    spec = importlib.util.spec_from_file_location("original_dsv41_compression", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    method, reason = CASES[case_id]
    _run_required_case(module.TestFusedLowRatioCompress, method, reason, source)
