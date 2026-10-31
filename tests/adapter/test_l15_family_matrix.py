"""CPU-only validation contracts for tools/l15_family_matrix.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "l15_family_matrix", ROOT / "tools" / "l15_family_matrix.py"
)
assert SPEC and SPEC.loader
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class L15FamilyMatrixTests(unittest.TestCase):
    def test_repository_matrix_is_valid_and_complete(self):
        document = json.loads(
            (ROOT / "tests" / "manifests" / "l15_family_matrix_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(matrix.validate(document), [])
        self.assertEqual(
            {family["id"] for family in document["families"]}, matrix.FAMILIES
        )
        self.assertGreaterEqual(
            sum(len(family["cases"]) for family in document["families"]), 20
        )

    def test_rejects_arbitrary_command_and_path_escape(self):
        document = json.loads(
            (ROOT / "tests" / "manifests" / "l15_family_matrix_v1.json").read_text(
                encoding="utf-8"
            )
        )
        bad = json.loads(json.dumps(document))
        bad["families"][0]["cases"][0] = {
            "id": "unsafe",
            "kind": "shell",
            "path": "../../escape",
        }
        errors = matrix.validate(bad)
        self.assertTrue(errors)
        self.assertIn("unsupported", errors[0])

    def test_requires_exact_upstream_nodeid(self):
        document = json.loads(
            (ROOT / "tests" / "manifests" / "l15_family_matrix_v1.json").read_text(
                encoding="utf-8"
            )
        )
        bad = json.loads(json.dumps(document))
        bad["families"][0]["cases"][0]["nodeid"] = (
            "test/registered/attention/unittests/dense/test_triton.py"
        )
        errors = matrix.validate(bad)
        self.assertTrue(errors)
        self.assertIn("exact nodeid", errors[0])


if __name__ == "__main__":
    unittest.main()
