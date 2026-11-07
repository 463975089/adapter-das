"""Padded EP input must never reach repeat_interleave with a false size."""

import hashlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from hcu_adapter import dsv41_request_indices as bridge

_SOURCE = """def token_req_indices(forward_batch, *, num_tokens=None):
    req = forward_batch.req_pool_indices.to(torch.int64)
    if forward_batch.forward_mode.is_decode():
        return req
    if forward_batch.forward_mode.is_target_verify():
        return torch.repeat_interleave(
            req, int(forward_batch.spec_info.draft_token_num), output_size=num_tokens)
    assert forward_batch.forward_mode.is_extend()
    return torch.repeat_interleave(
        req, forward_batch.extend_seq_lens.to(torch.int64), output_size=num_tokens)
"""


def _function(source=_SOURCE, filename="<request-indices>"):
    namespace = {"torch": torch}
    exec(compile(source, filename, "exec"), namespace)
    return namespace["token_req_indices"]


def _batch(mode="extend", requests=(16,), lengths=(6,)):
    return SimpleNamespace(
        req_pool_indices=torch.tensor(requests, dtype=torch.int32),
        extend_seq_lens=torch.tensor(lengths, dtype=torch.int32),
        forward_mode=SimpleNamespace(
            is_decode=lambda: mode == "decode",
            is_target_verify=lambda: mode == "verify",
            is_extend=lambda: mode == "extend",
        ),
        spec_info=SimpleNamespace(draft_token_num=3),
    )


def test_padded_alias_preserves_source_and_private_torch(tmp_path):
    source = tmp_path / "community.py"
    source.write_text(_SOURCE)
    before = hashlib.sha256(source.read_bytes()).digest()
    function = _function(filename=str(source))
    alias = function
    signature = inspect.signature(function)
    original_code = function.__code__
    bridge._install_function(function)
    assert bridge._install_function(function) is alias
    assert inspect.signature(alias) == signature
    assert getattr(alias, bridge._MARKER).__code__ is original_code
    assert function.__globals__["torch"] is torch
    assert alias(_batch(), num_tokens=8).tolist() == [16] * 6 + [0, 0]
    assert hashlib.sha256(source.read_bytes()).digest() == before
    assert [p.name for p in tmp_path.iterdir()] == ["community.py"]


@pytest.mark.parametrize("mode,size,expected", [
    ("decode", 2, [7, 9]),
    ("verify", 6, [7, 7, 7, 9, 9, 9]),
    ("extend", 5, [7, 7, 9, 9, 9]),
    ("extend", None, [7, 7, 9, 9, 9]),
])
def test_original_nonpadded_modes(mode, size, expected):
    function = _function()
    bridge._install_function(function)
    assert function(_batch(mode, (7, 9), (2, 3)), num_tokens=size).tolist() == expected


@pytest.mark.parametrize("change", ["extra-torch", "signature", "foreign"])
def test_upstream_drift_rejected_before_mutation(change):
    source = _SOURCE
    if change == "extra-torch":
        source = source.replace("    req =", "    torch.zeros(1)\n    req =")
    elif change == "signature":
        source = source.replace("num_tokens=None", "num_tokens=0")
    function = _function(source)
    if change == "foreign":
        function.__globals__[bridge._DISPATCH] = object()
    before = function.__code__
    with pytest.raises(RuntimeError):
        bridge._install_function(function)
    assert function.__code__ is before


def test_late_dispatch_drift_is_rejected():
    function = _function()
    bridge._install_function(function)
    function.__globals__[bridge._DISPATCH] = object()
    with pytest.raises(RuntimeError, match="changed after installation"):
        bridge._install_function(function)
