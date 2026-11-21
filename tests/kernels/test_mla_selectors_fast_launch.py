# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Fast-launch replay for the MLA selector kernels.

The replay path must be indistinguishable from the generic JIT dispatch: same
compiled kernel, same arguments, current stream, and a fallback on anything the
cached specialisation depends on. These tests qualify the replay on real HCU;
CPU cannot execute the kernels, so it only checks the gate bookkeeping.
"""
import pytest
import torch

from hcu_adapter.kernels import mla_selectors_triton as sel

SUPPORTED_ARCHES = ('gfx936', 'gfx938')


@pytest.fixture
def hcu():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('fast-launch replay requires HCU')
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    if arch not in SUPPORTED_ARCHES:
        pytest.skip('selector candidate requires gfx936/gfx938')
    return 'cuda'


@pytest.fixture(autouse=True)
def fresh_launchers(monkeypatch):
    """Opt in to replay on the original three-launch path."""
    monkeypatch.setenv(sel._FAST_LAUNCH_ENV, '1')
    for launcher in (sel._PREPARE_LAUNCH, sel._WRITE_LAUNCH):
        monkeypatch.setattr(launcher, '_sites', {})
        monkeypatch.setattr(launcher, '_off', None)
        monkeypatch.setattr(launcher, 'fallbacks', 0)
        monkeypatch.setattr(launcher, '_warned', False)


def _selectors(*args):
    return sel.selectors(*args, fused=False)


def _force_stock():
    """Disable the fast launch exactly like the kill switch does."""
    for launcher in (sel._PREPARE_LAUNCH, sel._WRITE_LAUNCH):
        launcher._off = True


def _csr(rows, width, device, offset=0, slots=31):
    big = torch.randint(0, slots, (rows * width + offset,), dtype=torch.int32,
                        device=device)
    flat = big[offset:]
    indptr = torch.arange(0, rows * width + 1, width, dtype=torch.int32, device=device)
    return None, None, flat, indptr


def _dense(rows, width, device):
    values = torch.randint(0, 31, (rows, 1, width), dtype=torch.int32, device=device)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    return values, lengths, None, None


def _stock(args, rows, slots=31):
    _force_stock()
    try:
        return _selectors(*args, rows, slots)
    finally:
        for launcher in (sel._PREPARE_LAUNCH, sel._WRITE_LAUNCH):
            launcher._off = None


@pytest.mark.parametrize('mode', ['csr', 'dense'])
@pytest.mark.parametrize('width', [1, 65, 129, 192, 256, 4096])
def test_replay_matches_stock(hcu, mode, width):
    rows = 6
    args = _csr(rows, width, hcu) if mode == 'csr' else _dense(rows, width, hcu)
    first = _selectors(*args, rows, 31)          # captures both kernels
    second = _selectors(*args, rows, 31)         # replays both kernels
    stock = _stock(args, rows)
    torch.cuda.synchronize()
    assert first[0].equal(stock[0]) and first[1].equal(stock[1])
    assert second[0].equal(stock[0]) and second[1].equal(stock[1])
    assert sel._PREPARE_LAUNCH._sites and sel._WRITE_LAUNCH._sites
    assert sel._PREPARE_LAUNCH.fallbacks == 0


def test_replay_keeps_counts_and_padding_of_duplicate_free_rows(hcu):
    """Guards the eager-sort contract that the two kernels wrap."""
    rows, width = 3, 192
    values = torch.full((rows, 1, width), -1, dtype=torch.int32, device=hcu)
    values[0, 0, 0], values[0, 0, 5] = 7, 7
    lengths = torch.tensor([width, 0, 2], dtype=torch.int32, device=hcu)
    args = (values, lengths, None, None)
    _selectors(*args, rows, 31)
    padded, counts = _selectors(*args, rows, 31)
    stock = _stock(args, rows)
    assert padded.equal(stock[0]) and counts.equal(stock[1])
    assert counts.tolist() == [2, 0, 0]


def test_kill_switch_never_captures(hcu, monkeypatch):
    monkeypatch.setenv(sel._FAST_LAUNCH_ENV, '0')
    args = _csr(4, 129, hcu)
    out = _selectors(*args, 4, 31)
    out2 = _selectors(*args, 4, 31)
    stock = _stock(args, 4)
    assert not sel._PREPARE_LAUNCH._sites and not sel._WRITE_LAUNCH._sites
    assert out[0].equal(stock[0]) and out2[0].equal(stock[0])


def test_batch_size_change_replays_same_site(hcu):
    """The batch size is a launch grid: it must not invalidate the site."""
    small = _csr(4, 129, hcu)
    _selectors(*small, 4, 31)
    sites = len(sel._PREPARE_LAUNCH._sites)
    big = _csr(16, 129, hcu)
    out = _selectors(*big, 16, 31)
    stock = _stock(big, 16)
    assert out[0].equal(stock[0]) and out[1].equal(stock[1])
    # _write_padded takes no shape argument, so every batch size replays the
    # same site; only the CSR capacity key of _prepare_selectors grows (<= bound).
    assert len(sel._WRITE_LAUNCH._sites) == 1
    assert len(sel._PREPARE_LAUNCH._sites) <= sel._PREPARE_LAUNCH._MAX_SITES
    assert sel._WRITE_LAUNCH.fallbacks == 0
    assert sites >= 1


def test_site_bound_falls_back(hcu, monkeypatch):
    """More concurrent capacities than sites falls back instead of thrashing."""
    monkeypatch.setattr(sel._FastLaunch, "_MAX_SITES", 3)
    width = 65
    for rows in range(1, 6):
        args = _csr(rows, width, hcu)
        out = _selectors(*args, rows, 31)
        stock = _stock(args, rows)
        assert out[0].equal(stock[0])
    assert len(sel._PREPARE_LAUNCH._sites) == 3
    assert sel._PREPARE_LAUNCH.fallbacks >= 1
    assert sel._PREPARE_LAUNCH._warned is True


def test_width_change_arms_new_site(hcu):
    """A width change is a new constexpr specialisation, not a broken site."""
    args = _csr(6, 129, hcu)
    _selectors(*args, 6, 31)
    wide = _csr(6, 192, hcu)
    out = _selectors(*wide, 6, 31)
    stock = _stock(wide, 6)
    assert out[0].equal(stock[0]) and out[1].equal(stock[1])
    assert len(sel._PREPARE_LAUNCH._sites) == 2
    assert sel._PREPARE_LAUNCH.fallbacks == 0


def test_misaligned_input_arms_new_site(hcu):
    aligned = _csr(6, 129, hcu)
    _selectors(*aligned, 6, 31)
    assert sel._PREPARE_LAUNCH._sites
    # Offset the CSR buffer by one int32 so the tensor pointer is no longer
    # 16-byte aligned: the cached specialisation must not be replayed.
    for offset in (1, 2, 3):
        candidate = _csr(6, 129, hcu, offset=offset)
        if candidate[2].data_ptr() % sel._ALIGN == 0:
            continue
        sites = len(sel._PREPARE_LAUNCH._sites)
        out = _selectors(*candidate, 6, 31)
        stock = _stock(candidate, 6)
        assert out[0].equal(stock[0]) and out[1].equal(stock[1])
        # The alignment differs from the captured site, so it must not replay.
        assert len(sel._PREPARE_LAUNCH._sites) == sites + 1
        return
    pytest.skip('could not build a misaligned CSR buffer on this allocator')


def test_pre_run_hooks_bypass_replay(hcu):
    args = _csr(4, 65, hcu)
    hook = sel._prepare_selectors.pre_run_hooks.append
    hook(lambda *a, **k: None)
    try:
        out = _selectors(*args, 4, 31)
    finally:
        sel._prepare_selectors.pre_run_hooks.pop()
    stock = _stock(args, 4)
    assert out[0].equal(stock[0])
    assert not sel._PREPARE_LAUNCH._sites


def test_replay_on_side_stream(hcu):
    args = _csr(6, 192, hcu)
    _selectors(*args, 6, 31)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = _selectors(*args, 6, 31)
    stream.synchronize()
    stock = _stock(args, 6)
    assert out[0].equal(stock[0]) and out[1].equal(stock[1])


def test_empty_width_path_unchanged(hcu):
    args = _dense_zero_width(hcu)
    out = _selectors(*args, 5, 31)
    stock = _stock(args, 5)
    assert out[0].equal(stock[0]) and out[1].equal(stock[1])
    assert not sel._PREPARE_LAUNCH._sites


def _dense_zero_width(device):
    values = torch.zeros((5, 1, 0), dtype=torch.int32, device=device)
    lengths = torch.zeros((5,), dtype=torch.int32, device=device)
    return values, lengths, None, None