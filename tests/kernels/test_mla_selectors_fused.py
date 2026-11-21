# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Fused single-kernel selector preparation (sglang-das 2026-09..11 port).

The window's HCU PRs collapse a torch-side top-k/sort + fill sequence into one
op (``f65889360f`` ragged top-k transform, ``da3925587d`` compact top-k).  The
adapter-side equivalent is: ``_prepare_selectors`` -> ``selected.sort`` ->
``_write_padded`` becomes one program that masks, sorts ascending and writes
the padded row.  The output must stay bit-identical, and shapes outside the
measured envelope must still take the three-step path.
"""
import pytest
import torch

from hcu_adapter.kernels import mla_selectors_triton as sel

SUPPORTED_ARCHES = ('gfx936', 'gfx938')


@pytest.fixture
def hcu():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('fused selectors require HCU')
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    if arch not in SUPPORTED_ARCHES:
        pytest.skip('selector candidate requires gfx936/gfx938')
    return 'cuda'


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Exercise generic JIT dispatch unless a test opts in to replay."""
    monkeypatch.setenv(sel._FAST_LAUNCH_ENV, '0')
    for launcher in (sel._PREPARE_LAUNCH, sel._WRITE_LAUNCH, sel._FUSED_LAUNCH):
        monkeypatch.setattr(launcher, '_sites', {})
        monkeypatch.setattr(launcher, '_off', None)
        monkeypatch.setattr(launcher, 'fallbacks', 0)
        monkeypatch.setattr(launcher, '_warned', False)


def _three_step(args, rows, slots=31):
    """Run the original three-launch path with the JIT dispatch untouched."""
    return sel.selectors(*args, rows, slots, fused=False)


def _csr(rows, width, device, offset=0, slots=31):
    big = torch.randint(0, slots, (rows * width + offset,), dtype=torch.int32,
                        device=device)
    flat = big[offset:]
    indptr = torch.arange(0, rows * width + 1, width, dtype=torch.int32,
                          device=device)
    return None, None, flat, indptr


def _dense(rows, width, device, slots=31):
    values = torch.randint(0, slots, (rows, 1, width), dtype=torch.int32,
                           device=device)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    return values, lengths, None, None


def _dirty(kind, rows, width, device, slots=31):
    """Duplicates, negatives and out-of-slot ids, all of which must pad out."""
    args = _csr(rows, width, device, slots=slots) if kind == 'csr' \
        else _dense(rows, width, device, slots=slots)
    source = args[2] if args[2] is not None else args[0]
    flat = source.reshape(-1)
    flat[::7] = -1
    flat[3::17] = slots + 5
    if flat.numel() > 4:
        flat[1] = flat[0]          # duplicate id
    return args


@pytest.mark.parametrize('mode', ['csr', 'dense'])
@pytest.mark.parametrize('rows,width', [(1, 1), (1, 65), (1, 129), (1, 192),
                                        (1, 512), (8, 128), (32, 128), (64, 128),
                                        (1, 1024), (1, 4096)])
def test_fused_matches_three_step(hcu, mode, rows, width):
    args = _csr(rows, width, hcu) if mode == 'csr' else _dense(rows, width, hcu)
    ref_ids, ref_lens = _three_step(args, rows)
    got_ids, got_lens = sel.selectors(*args, rows, 31)
    assert got_ids.equal(ref_ids)
    assert got_lens.equal(ref_lens)


@pytest.mark.parametrize('width', [4, 129, 512])
def test_dirty_input_matches_three_step(hcu, width):
    args = _dirty('csr', 9, width, hcu)
    ref_ids, ref_lens = _three_step(args, 9)
    got_ids, got_lens = sel.selectors(*args, 9, 31)
    assert got_ids.equal(ref_ids)
    assert got_lens.equal(ref_lens)
    assert int(ref_lens.min()) < width          # padding really was exercised


def test_zero_width(hcu):
    rows = 5
    flat = torch.zeros((0,), dtype=torch.int32, device=hcu)
    indptr = torch.zeros((rows + 1,), dtype=torch.int32, device=hcu)
    ref_ids, ref_lens = _three_step((None, None, flat, indptr), rows)
    got_ids, got_lens = sel.selectors(None, None, flat, indptr, rows, 31)
    assert got_ids.equal(ref_ids) and got_lens.equal(ref_lens)
    assert got_ids.shape == (rows, 1, 64)


@pytest.mark.parametrize('rows,width', [(1, 8192), (128, 4096)])
def test_shape_outside_envelope_falls_back(hcu, rows, width, monkeypatch):
    # Test both the direct helper's BLOCK limit and the production row budget.
    args = _dense(rows, width, hcu)
    ref = _three_step(args, rows)

    def unexpected_fusion(*args):
        pytest.fail('unexpected fused launch outside the shape budget')

    monkeypatch.setattr(sel, '_FUSED_LAUNCH', unexpected_fusion)
    out = sel.selectors(*args, rows, 31)
    assert out[0].equal(ref[0]) and out[1].equal(ref[1])


def test_fused_replays_after_first_call(hcu, monkeypatch):
    monkeypatch.setenv(sel._FAST_LAUNCH_ENV, '1')
    args = _csr(2, 192, hcu)
    first = sel.selectors(*args, 2, 31)
    second = sel.selectors(*args, 2, 31)
    assert first[0].equal(second[0]) and first[1].equal(second[1])
    assert len(sel._FUSED_LAUNCH._sites) == 1
    assert sel._FUSED_LAUNCH.fallbacks == 0
    assert not sel._PREPARE_LAUNCH._sites     # three-step path untouched