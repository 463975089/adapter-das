"""Byte-binding regressions for checkpoint-local code; no model execution."""

import hashlib

import pytest

from tools.checkpoint_files import audit, fingerprint


@pytest.mark.parametrize("relative", ["configuration_fixture.py", "helpers/model_math.py"])
@pytest.mark.parametrize("change", ["edit", "add", "remove"])
def test_checkpoint_python_changes_invalidate_existing_binding(tmp_path, relative, change):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"fixture weights, not a real model")
    source = tmp_path / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    original = b'raise AssertionError("fingerprinting must not execute this code")\n'
    if change != "add":
        source.write_bytes(original)
    before = fingerprint(tmp_path)
    if change == "add":
        source.write_bytes(original)
    else:
        assert before[relative] == hashlib.sha256(original).hexdigest()
        if change == "edit":
            # Same size; the byte binding cannot rely on file length.
            source.write_bytes(original.replace(b"AssertionError", b"ReferenceError"))
        else:
            source.unlink()
    result = audit(tmp_path, before)
    assert result["passed"] is False
    assert result["files_sha256"] != before
