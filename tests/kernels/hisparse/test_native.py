# SPDX-License-Identifier: Apache-2.0
"""REAL HIP numerical tests. Written for a later, explicitly authorized GPU lane.

Collection never queries/initializes a GPU; running tests does allocate device
memory. CPU reference is independent Python/list logic, not the native host model
or a production fallback. A requires-HIP skip is NOT a correctness result.
"""
from copy import deepcopy

import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(torch.version.hip is None, reason='requires HIP Torch and a free HCU')


@pytest.fixture(scope='module')
def native(request):
    if not torch.cuda.is_available():
        pytest.skip('requires an explicitly available HIP device; NOT validated')
    from hcu_adapter.kernels.hisparse import ensure_hisparse_ops
    ensure_hisparse_ops(request.config.getoption('--hisparse-library'))
    device = request.config.getoption('--hisparse-device')
    with torch.cuda.device(device):
        yield torch.ops._C_cache_ops


def _gpu(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype, device='cuda')


def _pinned(rows, width, offset):
    # Retain backing storage throughout the async launch and synchronization.
    owner = torch.empty(rows * width + offset + 31, dtype=torch.uint8, pin_memory=True)
    owner.copy_((torch.arange(owner.numel(), dtype=torch.int64) * 17 % 251).to(torch.uint8))
    return owner, owner[offset:offset + rows * width].view(rows, width)


def _hot(blocks, block_size, width, *, gap=24, offset=7):
    pitch = block_size * width + gap
    nbytes = offset + blocks * pitch + 17
    owner = torch.full((nbytes,), 85, dtype=torch.uint8, device='cuda')
    view = owner.as_strided((blocks, block_size, width), (pitch, width, 1), offset)
    reference = torch.full((nbytes,), 85, dtype=torch.uint8)
    ref_view = reference.as_strided(view.shape, view.stride(), offset)
    return owner, view, reference, ref_view


def _equal(gpu, cpu):
    expected = torch.as_tensor(cpu, dtype=gpu.dtype)
    torch.testing.assert_close(gpu.cpu(), expected, rtol=0, atol=0)


def _resolve_reference(indices, hot_table, dgi, lru, *, host_rows, requests,
                       states, source, resident, block_size, attention_stride,
                       stats, source_block_size=None, resident_block_size=None):
    """Independent request union, oldest-first eviction and stable list filter."""
    source_size = block_size if source_block_size is None else source_block_size
    resident_size = block_size if resident_block_size is None else resident_block_size
    rows, topk = len(indices), len(indices[0])
    hot = [[-1] * topk for _ in indices]
    attention = deepcopy(hot)
    resolved = deepcopy(hot)
    mask = [[0] * topk for _ in indices]
    sh = [[-77] * topk for _ in indices]
    sd = deepcopy(sh)
    counts = [0] * rows
    swap_counts = [0] * rows
    first = 0
    while first < rows:
        req = requests[first] if requests is not None else first
        end = first + 1
        if requests is not None:
            while end < rows and requests[end] == req:
                end += 1
        state = states[req] if states is not None else req
        if state < 0:
            first = end
            continue
        refs, keys = [], []
        for row in range(first, end):
            for col, token in enumerate(indices[row]):
                g, resident_row = token, -1
                if source is not None:
                    page = token // source_size if token >= 0 else -1
                    physical = source[req][page] if 0 <= page < len(source[req]) else 0
                    g = physical * source_size + token % source_size if physical > 0 else -1
                    resident_page = token // resident_size if token >= 0 else -1
                    if resident is not None and 0 <= resident_page < len(resident[state]):
                        physical = resident[state][resident_page]
                        if physical >= 0 and physical != 0:
                            resident_row = physical * resident_size + token % resident_size
                if g >= host_rows:
                    g = -1
                resolved[row][col] = g
                if resident_row >= 0:
                    hot[row][col] = resident_row
                elif g >= 0:
                    refs.append((row, col, g))
                    if g not in keys:
                        keys.append(g)
        if keys:
            hit_slots = [slot for slot in lru[state] if dgi[state][slot] in keys]
            evict = [slot for slot in lru[state] if slot not in hit_slots]
            slots = {dgi[state][slot]: slot for slot in hit_slots}
            missing = [g for g in keys if g not in slots]
            assert len(missing) <= len(evict)
            swaps = []
            for g, slot in zip(missing, evict):
                slots[g] = slot
                dgi[state][slot] = g
                physical = hot_table[req][slot // block_size] * block_size + slot % block_size
                swaps.append((g, physical))
            seen = set()
            for row, col, g in refs:
                slot = slots[g]
                physical = hot_table[req][slot // block_size] * block_size + slot % block_size
                hot[row][col] = physical
                if g in missing and g not in seen:
                    mask[row][col] = 1
                    stats[1] += 1
                else:
                    stats[0] += 1
                seen.add(g)
            for n, (g, physical) in enumerate(swaps):
                row, col = first + n // topk, n % topk
                sh[row][col], sd[row][col] = g, physical
                swap_counts[row] += 1
            lru[state] = evict[len(missing):] + evict[:len(missing)] + hit_slots
        for row in range(first, end):
            values = [(p // block_size) * attention_stride + p % block_size
                      for p in hot[row] if p >= 0]
            counts[row] = len(values)
            attention[row] = values + [-1] * (topk - len(values))
        first = end
    return dict(hot=hot, attention=attention, resolved=resolved, mask=mask,
                sh=sh, sd=sd, counts=counts, swaps=swap_counts)


def _case(name):
    config = dict(indices=[[-1, 2, 4, 4]], requests=[0], states=[0],
                  source=[[5, 6, 7, 8]], resident=None,
                  hot_table=[[1, 3]], dgi=[[10, 11, 12, 13]], lru=[[0, 1, 2, 3]])
    if name == 'multi_request':
        config.update(indices=[[-1, 0, 4, 4], [5, 1, -1, 4], [-1, 2, 4, 4]],
                      requests=[1, 1, 0], states=[1, 0],
                      source=[[5, 6, 7, 8], [9, 10, 11, 12]],
                      hot_table=[[1, 3], [4, 6]],
                      dgi=[[18, 19, 20, 21], [10, 11, 12, 13]],
                      lru=[[0, 1, 2, 3], [0, 1, 2, 3]])
    elif name == 'resident_only':
        config.update(indices=[[-1, 0, -1, 1]], resident=[[2, 0, 4, 0]])
    elif name == 'mixed_resident':
        config.update(indices=[[0, -1, 2, 4]], resident=[[2, 0, 0, 0]])
    elif name == 'padding':
        config.update(indices=[[-1, 0, 1, 2]], states=[-1])
    elif name == 'global_indices':
        config.update(indices=[[-1, 12, 14, 14]], requests=None, states=None, source=None)
    return config


def _run_resolve(native, name, width, *, rewrite=False):
    config = _case(name)
    indices = deepcopy(config['indices'])
    dgi_ref, lru_ref = deepcopy(config['dgi']), deepcopy(config['lru'])
    rows, topk = len(indices), len(indices[0])
    owner, host = _pinned(64, width, offset=19)
    hot_owner, hot, expected_storage, expected_hot = _hot(16, 2, width)
    dgi, lru = _gpu(dgi_ref), _gpu(lru_ref, torch.int16)
    logical, hot_table = _gpu(indices), _gpu(config['hot_table'])
    requests = None if config['requests'] is None else _gpu(config['requests'])
    states = None if config['states'] is None else _gpu(config['states'])
    source = None if config['source'] is None else _gpu(config['source'])
    resident = None if config['resident'] is None else _gpu(config['resident'])
    stats, stats_ref = torch.zeros(2, dtype=torch.uint64, device='cuda'), [0, 0]
    # Counts and attention outputs exercise non-unit row/count strides.
    attention_storage = torch.full((rows, topk + 3), -99, dtype=torch.int32, device='cuda')
    attention = attention_storage[:, :topk]
    count_storage = torch.full((rows * 2,), -99, dtype=torch.int32, device='cuda')
    counts = count_storage[::2]
    # Start with consistent ownership/payloads, not arbitrary cached bytes.
    for req, table in enumerate(config['hot_table']):
        state = config['states'][req] if config['states'] is not None else req
        if state < 0:
            continue
        for slot, g in enumerate(dgi_ref[state]):
            physical = table[slot // 2] * 2 + slot % 2
            expected_hot[physical // 2, physical % 2].copy_(host[g])
        if config['resident'] is not None:
            for page, physical_block in enumerate(config['resident'][state]):
                if physical_block > 0:
                    for lane in range(2):
                        physical = physical_block * 2 + lane
                        g = config['source'][req][page] * 2 + lane
                        expected_hot[physical // 2, physical % 2].copy_(host[g])
    hot.copy_(expected_hot.to(device='cuda'))
    # Changed input replay here means repeated eager calls, NOT a captured Graph.
    steps = [indices, deepcopy(indices)]
    if name == 'holes_eviction' and not rewrite:
        steps[1] = [[0, 1, 4, -1]]
    for step, values in enumerate(steps):
        if step and rewrite:
            # The previous stream has completed before the CPU writer mutates
            # mapped storage. Rewriting a cached row must force a new miss.
            host[14].fill_(37)
            native.hisparse_invalidate_written_slots(dgi, states, _gpu([0]), _gpu([14], torch.int64))
            dgi_ref[0] = [-1 if g == 14 else g for g in dgi_ref[0]]
        logical.copy_(_gpu(values))
        outputs = {k: torch.full((rows, topk), -77, dtype=torch.int32, device='cuda')
                   for k in ('hot', 'mask', 'resolved', 'sh', 'sd')}
        swaps = torch.full((rows,), -77, dtype=torch.int32, device='cuda')
        expected = _resolve_reference(values, config['hot_table'], dgi_ref, lru_ref,
                    host_rows=64, requests=config['requests'], states=config['states'],
                    source=config['source'], resident=config['resident'], block_size=2,
                    attention_stride=5, stats=stats_ref)
        native.hisparse_resolve_residency(
            host, hot, hot_table, logical, outputs['hot'], dgi, lru, states, 4, 4,
            outputs['mask'], stats, attention, 5, requests, source, 2 if source is not None else 0,
            outputs['resolved'], counts, outputs['sh'], outputs['sd'], swaps,
            resident, 2 if resident is not None else 0, 0)
        native.hisparse_gather_compact(host, hot, outputs['sh'], outputs['sd'], swaps)
        for row, count in enumerate(expected['swaps']):
            for col in range(count):
                physical = expected['sd'][row][col]
                expected_hot[physical // 2, physical % 2].copy_(host[expected['sh'][row][col]])
        torch.cuda.synchronize()
        for key, value in outputs.items():
            _equal(value, expected[key])
        _equal(swaps, expected['swaps'])
        _equal(attention, expected['attention'])
        _equal(counts, expected['counts'])
        _equal(dgi, dgi_ref)
        _equal(lru, lru_ref)
        _equal(stats, stats_ref)
        _equal(hot_owner, expected_storage)
        _equal(attention_storage[:, topk:], [[-99] * 3 for _ in range(rows)])
        _equal(count_storage[1::2], [-99] * rows)
    assert owner.numel() >= host.numel()  # backing lifetime retained through every launch


@pytest.mark.parametrize('name', ['holes_eviction', 'multi_request', 'resident_only',
                                  'mixed_resident', 'padding', 'global_indices'])
@pytest.mark.parametrize('width', [16, 528])
def test_resolve_state_and_compact_reference(native, name, width):
    _run_resolve(native, name, width)


def test_rewrite_invalidates_cached_row_and_recopies_new_bytes(native):
    _run_resolve(native, 'holes_eviction', 64, rewrite=True)


def test_invalidate_all_duplicate_slots_and_invalid_requests(native):
    values = [[18, 19, 18, -1], [10, 11, 10, 12]]
    dgi = _gpu(values)
    states, requests, written = [1, 0, -1], [0, 1, -1, 2, 7, 1], [10, 18, 11, 20, 22, 18]
    expected = deepcopy(values)
    for req, slot in zip(requests, written):
        if 0 <= req < len(states) and 0 <= states[req] < len(expected) and slot >= 0:
            expected[states[req]] = [-1 if value == slot else value for value in expected[states[req]]]
    native.hisparse_invalidate_written_slots(dgi, _gpu(states), _gpu(requests), _gpu(written, torch.int64))
    torch.cuda.synchronize()
    _equal(dgi, expected)


@pytest.mark.parametrize('offset', [0, 3, 19])
def test_gather_plan_pinned_interior_offset_and_page_pitch(native, offset):
    owner, host = _pinned(8, 528, offset)
    hot_owner, hot, expected, expected_hot = _hot(6, 2, 528)
    globals_ = [[-1, 2, 99, 3, 4], [1, 2, 3, 4, 5]]
    destinations = [[4, 2, 3, 8, -1], [4, 5, 6, 7, 8]]
    masks, state = [[1, 1, 1, 0, 1], [1] * 5], [0, -1]
    attention = torch.empty((2, 5), dtype=torch.int32, device='cuda')
    ref_attention = []
    for row in range(2):
        ref_attention.append([])
        for g, dst, miss in zip(globals_[row], destinations[row], masks[row]):
            ref_attention[row].append(-1 if state[row] < 0 or dst < 0 else dst // 2 * 5 + dst % 2)
            if state[row] >= 0 and miss and g >= 0 and 0 <= dst < 12:
                if g < len(host):
                    expected_hot[dst // 2, dst % 2].copy_(host[g])
                else:
                    expected_hot[dst // 2, dst % 2].zero_()
    native.hisparse_gather_plan(host, hot, _gpu(globals_), _gpu(destinations), _gpu(masks), _gpu(state), attention, 5)
    torch.cuda.synchronize()
    _equal(hot_owner, expected)
    _equal(attention, ref_attention)
    assert owner.numel() > host.numel()


def test_gather_compact_clamps_counts_and_zeros_missing_host_rows(native):
    owner, host = _pinned(8, 64, 3)
    hot_owner, hot, expected, expected_hot = _hot(6, 2, 64)
    globals_ = [[2, 99, -1, 3], [3, 4, 5, 6], [0, 1, 2, 3]]
    destinations = [[2, 3, 4, 5], [2, 3, 4, 5], [6, 7, 8, 9]]
    counts = [2, -1, 99]
    for row, count in enumerate(counts):
        for col in range(max(0, min(count, 4))):
            g, dst = globals_[row][col], destinations[row][col]
            if g >= 0 and 0 <= dst < 12:
                if g < len(host):
                    expected_hot[dst // 2, dst % 2].copy_(host[g])
                else:
                    expected_hot[dst // 2, dst % 2].zero_()
    native.hisparse_gather_compact(host, hot, _gpu(globals_), _gpu(destinations), _gpu(counts))
    torch.cuda.synchronize()
    _equal(hot_owner, expected)
    assert owner.is_pinned()


def test_gather_plan_supports_dense_2d_staging(native):
    owner, host = _pinned(8, 64, 19)
    staged = torch.full((4, 64), 85, dtype=torch.uint8, device='cuda')
    attention = torch.empty((1, 3), dtype=torch.int32, device='cuda')
    native.hisparse_gather_plan(host, staged, _gpu([[2, 3, -1]]), _gpu([[0, 1, 2]]),
                                _gpu([[1, 1, 1]]), None, attention, 1)
    expected = torch.full((4, 64), 85, dtype=torch.uint8)
    expected[:2].copy_(host[2:4])
    torch.cuda.synchronize()
    _equal(staged, expected)
    _equal(attention, [[0, 1, 2]])
    assert owner.is_pinned()


def test_empty_launches_do_not_modify_cache_or_state(native):
    owner, host = _pinned(8, 16, 3)
    hot_owner, hot, expected, _ = _hot(6, 2, 16)
    empty_indices = torch.empty((0, 4), dtype=torch.int32, device='cuda')
    empty_rows = torch.empty((0,), dtype=torch.int32, device='cuda')
    native.hisparse_gather_compact(host, hot, empty_indices, empty_indices, empty_rows)
    native.hisparse_gather_plan(host, hot, empty_indices, empty_indices, empty_indices, None)
    dgi = _gpu([[10, 11, 12, 13]])
    native.hisparse_invalidate_written_slots(dgi, _gpu([0]), empty_rows,
                                           torch.empty((0,), dtype=torch.int64, device='cuda'))
    native.hisparse_resolve_residency(host, hot, _gpu([[1, 2]]), empty_indices,
        empty_indices, dgi, _gpu([[0, 1, 2, 3]], torch.int16), _gpu([0]), 4, 4)
    torch.cuda.synchronize()
    _equal(hot_owner, expected)
    _equal(dgi, [[10, 11, 12, 13]])
    assert owner.is_pinned()


def test_pageable_host_and_bad_plan_dtype_refused(native):
    owner, host = _pinned(8, 64, 3)
    hot_owner, hot, expected, _ = _hot(6, 2, 64)
    with pytest.raises(RuntimeError, match='pinned'):
        native.hisparse_gather_compact(host.clone(), hot, _gpu([[2]]), _gpu([[2]]), _gpu([1]))
    with pytest.raises(RuntimeError, match='int32'):
        native.hisparse_gather_compact(host, hot, _gpu([[2]], torch.int64), _gpu([[2]]), _gpu([1]))
    torch.cuda.synchronize()
    _equal(hot_owner, expected)
    assert owner.is_pinned()


def test_attention_address_overflow_refused(native):
    owner, host = _pinned(8, 16, 3)
    hot_owner, hot, expected, _ = _hot(6, 2, 16)
    indices, hots = _gpu([[2]]), _gpu([[2]])
    attention = torch.empty((1, 1), dtype=torch.int32, device='cuda')
    with pytest.raises(RuntimeError, match='fit int32'):
        native.hisparse_gather_plan(host, hot, indices, hots, _gpu([[1]]),
                                   None, attention, 2**63 - 1)
    with pytest.raises(RuntimeError, match='fit int32'):
        native.hisparse_resolve_residency(host, hot, _gpu([[1, 2]]), indices,
            hots, _gpu([[0, 1, 2, 3]]), _gpu([[0, 1, 2, 3]], torch.int16),
            _gpu([0]), 4, 4, attention_indices=attention,
            attention_block_stride=2**63 - 1)
    torch.cuda.synchronize()
    _equal(hot_owner, expected)
    assert owner.is_pinned()


@pytest.mark.parametrize('physical_block', [64, 256])
def test_resolved_kernel256_preserves_state64(native, physical_block):
    # Scheduler/source addressing is 256, independently of the 64-slot state.
    # Exercise both a separate 64-row hot pool and a 256-row physical page.
    owner, host = _pinned(1024, 16, offset=19)
    hot_owner, hot, expected_storage, expected_hot = _hot(4, physical_block, 16)
    indices = [[63, -1, 64, 64, 65]]
    dgi_ref, lru_ref = [list(range(256, 320))], [list(range(64))]
    for slot, g in enumerate(dgi_ref[0]):
        expected_hot[1, slot].copy_(host[g])
    hot.copy_(expected_hot.to(device='cuda'))
    dgi, lru = _gpu(dgi_ref), _gpu(lru_ref, torch.int16)
    outputs = {k: torch.full((1, 5), -77, dtype=torch.int32, device='cuda')
               for k in ('hot', 'mask', 'resolved', 'sh', 'sd')}
    attention = torch.full((1, 5), -77, dtype=torch.int32, device='cuda')
    counts, swaps = _gpu([-77]), _gpu([-77])
    stats = torch.zeros(2, dtype=torch.uint64, device='cuda')
    stats_ref = [0, 0]
    expected = _resolve_reference(indices, [[1]], dgi_ref, lru_ref,
        host_rows=1024, requests=[0], states=[0], source=[[1, 2]], resident=None,
        block_size=physical_block, attention_stride=256, stats=stats_ref,
        source_block_size=256)
    # Literal known answers prevent the oracle from silently retaining 64.
    assert expected['resolved'] == [[319, -1, 320, 320, 321]]
    assert expected['attention'] == [[319, 256, 256, 257, -1]]
    native.hisparse_resolve_residency(host, hot, _gpu([[1]]), _gpu(indices),
        outputs['hot'], dgi, lru, _gpu([0]), 64, 64, outputs['mask'], stats,
        attention, 256, _gpu([0]), _gpu([[1, 2]]), 256,
        outputs['resolved'], counts, outputs['sh'], outputs['sd'], swaps)
    native.hisparse_gather_compact(host, hot, outputs['sh'], outputs['sd'], swaps)
    for col in range(expected['swaps'][0]):
        physical = expected['sd'][0][col]
        expected_hot[physical // physical_block, physical % physical_block].copy_(
            host[expected['sh'][0][col]])
    torch.cuda.synchronize()
    for key, value in outputs.items():
        _equal(value, expected[key])
    for actual, value in [(attention, expected['attention']), (counts, expected['counts']),
                          (swaps, expected['swaps']), (dgi, dgi_ref), (lru, lru_ref),
                          (stats, stats_ref), (hot_owner, expected_storage)]:
        _equal(actual, value)
    assert owner.is_pinned()


@pytest.mark.parametrize('block_size', [64, 128, 256])
@pytest.mark.parametrize('copy_kind', ['plan', 'compact'])
@pytest.mark.parametrize('width', [16, 528])
def test_copy_block_metadata_above64(native, block_size, copy_kind, width):
    owner, host = _pinned(8, width, offset=19)
    hot_owner, hot, expected_storage, expected_hot = _hot(4, block_size, width)
    destinations = [block_size - 1, block_size, block_size + min(63, block_size - 2),
                    2 * block_size - 1, 2 * block_size, 2 * block_size + 64,
                    3 * block_size - 1]
    assert len(destinations) == len(set(destinations))  # No cross-warp write races.
    globals_ = [0, 1, 2, 99, 3, 4, 5]
    for g, dst in zip(globals_, destinations):
        row = expected_hot[dst // block_size, dst % block_size]
        if g < len(host):
            row.copy_(host[g])
        else:
            row.zero_()
    if copy_kind == 'plan':
        attention = torch.full((1, 7), -77, dtype=torch.int32, device='cuda')
        native.hisparse_gather_plan(host, hot, _gpu([globals_]),
            _gpu([destinations]), _gpu([[1] * 7]), None, attention, 256)
    else:
        native.hisparse_gather_compact(host, hot, _gpu([globals_]),
                                     _gpu([destinations]), _gpu([7]))
    torch.cuda.synchronize()
    _equal(hot_owner, expected_storage)  # Includes page gaps and guard bytes.
    if copy_kind == 'plan':
        _equal(attention, [[dst // block_size * 256 + dst % block_size
                            for dst in destinations]])
    assert owner.is_pinned()
