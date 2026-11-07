"""Negative controls for the original-test rehosting harness; no GPU required."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def harness():
    source = Path(__file__).parents[1] / "kernels/test_dsv41_compression.py"
    spec = importlib.util.spec_from_file_location("compression_harness_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("behavior", [
    "pass", "failure", "subtest-failure", "skip", "setup-failure",
    "expected-failure", "unexpected-success",
])
def test_required_case_keeps_outcomes_and_cleanup(harness, tmp_path, behavior):
    body = {
        "pass": "self.assertEqual(1, 1)",
        "failure": "self.assertEqual(1, 2)",
        "subtest-failure": "with self.subTest(part=1):\n            self.assertEqual(1, 2)",
        "skip": "self.skipTest('another required capability unavailable')",
        "setup-failure": "self.assertEqual(1, 1)",
        "expected-failure": "self.assertEqual(1, 2)",
        "unexpected-success": "self.assertEqual(1, 1)",
    }[behavior]
    source = tmp_path / "original_test.py"
    source.write_text(
        "import unittest\n"
        "events=[]\n"
        "def is_hip(): return False\n"
        "class OriginalCase(unittest.TestCase):\n"
        "    @classmethod\n"
        "    def setUpClass(cls):\n"
        "        cls.addClassCleanup(events.append, 'class-cleanup')\n"
        "    def setUp(self):\n"
        "        self.addCleanup(events.append, 'test-cleanup')\n"
        + ("        raise RuntimeError('setup failed')\n" if behavior == "setup-failure" else "")
        + "    @unittest.skipUnless(is_hip(), 'HIP fused compressor regression')\n"
        "    def test_original(self):\n"
        f"        {body}\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("original_fixture", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    decorated = original.OriginalCase.test_original
    if behavior in ("expected-failure", "unexpected-success"):
        decorated.__wrapped__.__unittest_expecting_failure__ = True
    run = lambda: harness._run_required_case(
        original.OriginalCase, "test_original", "HIP fused compressor regression", source)
    if behavior == "pass":
        run()
    else:
        with pytest.raises(AssertionError, match="incomplete or failed"):
            run()
    assert original.OriginalCase.test_original is decorated
    assert decorated.__unittest_skip__ is True
    assert original.events == ["test-cleanup", "class-cleanup"]


def test_required_case_rejects_changed_decorator(harness, tmp_path):
    source = tmp_path / "changed_test.py"
    source.write_text(
        "import unittest\n"
        "class OriginalCase(unittest.TestCase):\n"
        "    @unittest.skip('unsupported')\n"
        "    def test_original(self): raise AssertionError('must not execute')\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("changed_fixture", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    with pytest.raises(AssertionError, match="skip contract changed"):
        harness._run_required_case(
            original.OriginalCase, "test_original", "HIP fused compressor regression", source)
