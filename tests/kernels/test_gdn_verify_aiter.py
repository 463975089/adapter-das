# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""GDN AITER verify: FP32 recurrence, snapshot ABI and changing Graph inputs.

Run on an idle HCU: HCU_LINEAR_ATTN_GDN_VERIFY=aiter python -m pytest
    tests/kernels/test_gdn_verify_aiter.py -v
"""
import inspect
import math
import os
import subprocess
import sys
import types
import random

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU GPU required")
SENTINEL = -7.0


@pytest.fixture(scope="module")
def routed_face():
    if torch.version.hip is None:
        pytest.skip("DTK AITER route requires HIP")
    from hcu_adapter import gdn_aiter as bridge
    from sglang.kernels.ops.attention.fla import fused_sigmoid_gating_recurrent as face
    provider = pytest.importorskip("aiter.ops.triton.fla.fused_sigmoid_gating_recurrent")
    vendor = getattr(provider, "fused_sigmoid_gating_delta_rule_update", None)
    if vendor is None:
        pytest.skip("installed AITER has no GDN target-verify provider")
    current = face.fused_sigmoid_gating_delta_rule_update
    upstream = getattr(current, bridge._VERIFY_MARKER, current)
    namespace = dict(upstream.__globals__)
    namespace.pop(bridge._VERIFY_DISPATCH, None)
    # Numerical vendor contracts must not depend on the process-wide auto
    # selection (ragged/fixed variants intentionally lie outside auto scope).
    fn = types.FunctionType(upstream.__code__, namespace, upstream.__name__,
                            upstream.__defaults__, upstream.__closure__)
    fn.__kwdefaults__ = upstream.__kwdefaults__
    bridge._bind_target_verify_aiter(fn, vendor, torch, policy="aiter")
    return fn, upstream


def make_case(n, steps=4, kind="chain", seed=17):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    h, hv, d = 8, 24, 128
    total, slots = n * steps, n + 3
    qkv = torch.randn(1, total, 2 * h * d + hv * d, device="cuda",
                      dtype=torch.bfloat16) * 0.3
    q = qkv[..., :h*d].view(1, total, h, d)
    k = qkv[..., h*d:2*h*d].view(1, total, h, d)
    v = qkv[..., 2*h*d:].view(1, total, hv, d)
    ab_dtype = torch.bfloat16 if kind in ("bf16_gates", "sglang_2d", "sglang_2d_strided") else torch.float32
    ab = torch.randn(1, total, hv, 2, device="cuda", dtype=ab_dtype) * 0.2
    lengths = [steps] * n
    if kind == "ragged":
        lengths = [1 + i % steps for i in range(n)]
    if kind == "padded":
        lengths[-1] = 0
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    indices = rng.sample(range(slots), n)
    scratch_indices = rng.sample(range(slots), n)
    if kind == "no_state":
        indices[-1] = -1
    if kind == "padded":
        indices[-1] = scratch_indices[-1] = -1
    parents = [[-1] + list(range(steps-1)) for _ in range(n)]
    if kind in ("tree", "strided", "padded") and steps >= 3:
        for row in parents:
            row[2] = 0
    p = dict(
        A_log=torch.randn(hv, device="cuda", dtype=torch.float32) * 0.2,
        dt_bias=torch.randn(hv, device="cuda", dtype=torch.float32) * 0.2,
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, a=ab[..., 0], b=ab[..., 1],
        initial_state_source=torch.randn(slots, hv, d, d, device="cuda",
                                         dtype=torch.float32) * 0.1,
        initial_state_indices=torch.tensor(indices, device="cuda", dtype=torch.int32),
        scale=d**-0.5, use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.tensor(cu, device="cuda", dtype=torch.int32),
        is_kda=False, disable_state_update=True,
        intermediate_states_buffer=torch.full((slots, steps, hv, d, d), SENTINEL,
                                               device="cuda", dtype=torch.float32),
        intermediate_state_indices=torch.tensor(scratch_indices, device="cuda",
                                                 dtype=torch.int32),
        cache_steps=steps,
        retrieve_parent_token=torch.tensor(parents, device="cuda", dtype=torch.int32),
    )
    if kind in ("sglang_2d", "sglang_2d_strided"):
        p["a"], p["b"] = p["a"].view(total, hv), p["b"].view(total, hv)
        if kind == "sglang_2d":
            p["a"], p["b"] = p["a"].contiguous(), p["b"].contiguous()
        else:
            for name in ("initial_state_indices", "intermediate_state_indices",
                         "cu_seqlens", "retrieve_parent_token"):
                p[name] = p[name].long()
    if kind == "fixed":
        for name in ("q", "k", "v"):
            p[name] = p[name].view(n, steps, *p[name].shape[2:])
        p["a"], p["b"] = p["a"].view(n, steps, hv), p["b"].view(n, steps, hv)
        p["cu_seqlens"] = p["retrieve_parent_token"] = None
        p["use_qk_l2norm_in_kernel"] = False
    meta = dict(lengths=lengths, cu=cu, indices=indices, scratch=scratch_indices,
                parents=parents, used=cu[-1])
    return p, meta


def fp32_reference(p, meta):
    h, hv = p["q"].shape[-2], p["v"].shape[-2]
    q = p["q"].float().reshape(1, -1, h, 128)
    k = p["k"].float().reshape(1, -1, h, 128)
    v = p["v"].float().reshape(1, -1, hv, 128)
    a, b = p["a"].reshape(-1, hv), p["b"].reshape(-1, hv)
    if p["use_qk_l2norm_in_kernel"]:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q, k = q.repeat_interleave(hv // h, dim=2), k.repeat_interleave(hv // h, dim=2)
    out = torch.zeros_like(v, dtype=torch.float32)
    cache = torch.full_like(p["intermediate_states_buffer"], SENTINEL)
    written = []
    for row, length in enumerate(meta["lengths"]):
        idx = meta["indices"][row]
        state = (p["initial_state_source"][idx].float().transpose(-1, -2).clone()
                 if idx >= 0 else torch.zeros(hv, 128, 128, device="cuda"))
        history = []
        for t in range(length):
            if t:
                parent = (meta["parents"][row][t]
                          if p["retrieve_parent_token"] is not None
                          and meta["scratch"][row] >= 0 else t-1)
                state = history[parent].clone()
            pos = meta["cu"][row] + t
            x = a[pos].float() + p["dt_bias"]
            gate = torch.exp(-torch.exp(p["A_log"]) * F.softplus(
                x, beta=p["softplus_beta"], threshold=p["softplus_threshold"]))
            state = state * gate[:, None, None]
            key = k[0, pos]
            pred = (state * key[..., None]).sum(-2)
            delta = (v[0, pos] - pred) * torch.sigmoid(b[pos].float())[:, None]
            state = state + key[..., None] * delta[:, None, :]
            history.append(state.clone())
            out[0, pos] = (state * q[0, pos, :, :, None]).sum(-2) * p["scale"]
            slot = meta["scratch"][row]
            if slot >= 0:
                cache[slot, t] = state.transpose(-1, -2)
                written.append((slot, t))
    return out[:, :meta["used"]], cache, written


def assert_values(actual, expected, relative, label):
    a, b = actual.float().flatten(), expected.float().flatten()
    if b.numel() == 0:
        assert a.numel() == 0, label
        return
    maxref = b.abs().max().item()
    err = (a-b).abs().max().item()
    rel = err / max(maxref, 1e-12)
    cos = F.cosine_similarity(a, b, dim=0).item() if maxref else 1.0
    print(label, dict(cos=cos, maxabs=err, maxrel=rel), flush=True)
    assert cos >= 0.9999 and (rel <= relative or err <= 2e-7), (
        label, dict(cos=cos, maxabs=err, maxrel=rel))


def check_result(p, meta, out, ref_out, ref_cache, written, state_before):
    flat_out = out.reshape(1, -1, *out.shape[-2:])
    assert_values(flat_out[:, :meta["used"]], ref_out, 5e-2, "BF16 output vs FP32")
    got = p["intermediate_states_buffer"]
    assert torch.isfinite(got).all().item()
    if not written:
        assert meta["used"] == 0
        assert torch.equal(got, ref_cache)
        assert torch.equal(p["initial_state_source"], state_before)
        return
    chunks_a = torch.cat([got[i,t].flatten() for i,t in written])
    chunks_b = torch.cat([ref_cache[i,t].flatten() for i,t in written])
    assert_values(chunks_a, chunks_b, 1e-4, "FP32 snapshots vs independent recurrence")
    # A wrong index must also fail, even if the written values look accurate.
    denom = max(chunks_b.abs().max().item(), 1e-12)
    assert (got-ref_cache).abs().max().item() / denom <= 1e-4
    assert torch.equal(p["initial_state_source"], state_before)


@pytest.mark.parametrize("n,steps,kind", [
    (1,4,"chain"), (4,4,"tree"), (16,4,"strided"),
    (3,3,"ragged"), (4,4,"padded"), (17,4,"chain"),
    (3,4,"fixed"), (1,4,"no_state"), (4,4,"bf16_gates"),
    (18,4,"sglang_2d"), (16,4,"sglang_2d_strided"),
])
def test_verify_output_snapshot_and_source_contract(routed_face, n, steps, kind):
    fn, upstream = routed_face
    p, meta = make_case(n, steps, kind)
    ref_out, ref_cache, written = fp32_reference(p, meta)
    before = p["initial_state_source"].clone()
    from hcu_adapter.gdn_aiter import _verify_contract_supported
    assert _verify_contract_supported(p, torch), "fixture must exercise the AITER route"
    out = fn(**p)
    torch.cuda.synchronize()
    check_result(p, meta, out, ref_out, ref_cache, written, before)
    p["intermediate_states_buffer"].fill_(SENTINEL)
    # The upstream kernel assumes dense beta heads. Compare the same values
    # in its canonical gate layout, while testing the adapter with strided beta.
    base = upstream(**dict(p, a=p["a"].contiguous(), b=p["b"].contiguous()))
    torch.cuda.synchronize()
    assert_values(out[:, :meta["used"]], base[:, :meta["used"]], 5e-2,
                  "AITER vs original")
    check_result(p, meta, base, ref_out, ref_cache, written, before)


def test_verify_graph_replay_changes_values_indices_and_tree(routed_face, dt_bias_dtype=None, parents_enabled=True, n=16):
    fn, _ = routed_face
    p, meta = make_case(n, 4, "sglang_2d_strided")
    if dt_bias_dtype is not None:
        p["dt_bias"] = p["dt_bias"].to(dt_bias_dtype)
    if not parents_enabled:
        p["retrieve_parent_token"] = None
    from hcu_adapter.gdn_aiter import _verify_contract_supported
    assert _verify_contract_supported(p, torch), "Graph must capture the AITER route"
    for _ in range(2):
        fn(**p)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(**p)
    # Capture is not an execution. Establish a real replay baseline first.
    graph.replay()
    torch.cuda.synchronize()
    before = p["initial_state_source"].clone()
    ref_out, ref_cache, written = fp32_reference(p, meta)
    check_result(p, meta, output, ref_out, ref_cache, written, before)
    for seed, kind in [(31,"chain"), (37,"strided"), (43,"padded")]:
        fresh, meta = make_case(n, 4, kind, seed)
        fresh["a"] = fresh["a"].reshape(-1, 24).to(p["a"].dtype)
        fresh["b"] = fresh["b"].reshape(-1, 24).to(p["b"].dtype)
        for name, value in fresh.items():
            if (isinstance(value, torch.Tensor) and name != "intermediate_states_buffer"
                    and p[name] is not None):
                p[name].copy_(value)
        p["intermediate_states_buffer"].fill_(SENTINEL)
        before = p["initial_state_source"].clone()
        ref_out, ref_cache, written = fp32_reference(p, meta)
        graph.replay()
        torch.cuda.synchronize()
        check_result(p, meta, output, ref_out, ref_cache, written, before)


@pytest.fixture
def automatic_verify_face(routed_face):
    """Same official face/kernel globals, isolated binding under auto policy."""
    from hcu_adapter import gdn_aiter as bridge
    from aiter.ops.triton.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as vendor,
    )
    if not bridge._verify_auto_stack_supported():
        pytest.skip("GDN verify auto admission is limited to the qualified DTK2610 build")
    arch = torch.cuda.get_device_properties(torch.device("cuda", 0)).gcnArchName.split(":")[0]
    if arch not in bridge._VERIFY_AUTO_ARCHES:
        pytest.skip("GDN verify auto requires a qualified gfx936/gfx938 device")
    _, upstream = routed_face
    namespace = dict(upstream.__globals__)
    namespace.pop(bridge._VERIFY_DISPATCH, None)
    fn = types.FunctionType(upstream.__code__, namespace, upstream.__name__,
                            upstream.__defaults__, upstream.__closure__)
    fn.__kwdefaults__ = upstream.__kwdefaults__
    count = [0]
    def observed_vendor(**kwargs):
        count[0] += 1
        return vendor(**kwargs)
    observed_vendor.__signature__ = inspect.signature(vendor)
    observed_vendor.__module__ = vendor.__module__
    bridge._bind_target_verify_aiter(fn, observed_vendor, torch, policy="auto")
    return fn, upstream, count


@pytest.mark.parametrize("n", [1,4,8,16,18,19,32,33,63,64])
@pytest.mark.parametrize("parents_enabled", [True, False])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
def test_auto_target_verify_real_gate_layout(automatic_verify_face, bias_dtype, parents_enabled, n):
    fn, _, count = automatic_verify_face
    p, meta = make_case(n, 4, "sglang_2d")
    p["dt_bias"] = p["dt_bias"].to(bias_dtype)
    if not parents_enabled: p["retrieve_parent_token"] = None
    before = p["initial_state_source"].clone()
    ref_out, ref_cache, written = fp32_reference(p, meta)
    out = fn(**p)
    torch.cuda.synchronize()
    assert count[0] == 1, "qualified auto must execute AITER"
    check_result(p, meta, out, ref_out, ref_cache, written, before)


@pytest.mark.parametrize("n", [1,4,8,16,32,64])
@pytest.mark.parametrize("parents_enabled", [True, False])
def test_auto_target_verify_changing_input_graph(automatic_verify_face, parents_enabled, n):
    fn, upstream, count = automatic_verify_face
    test_verify_graph_replay_changes_values_indices_and_tree(
        (fn, upstream), dt_bias_dtype=torch.bfloat16, parents_enabled=parents_enabled, n=n)
    # Two eager warmups and capture; replay executes no Python dispatcher.
    assert count[0] == 3, "Graph must contain AITER rather than upstream fallback"


def test_auto_target_verify_unqualified_normalization_uses_upstream(automatic_verify_face):
    fn, _, count = automatic_verify_face
    p, meta = make_case(65, 4, "chain")
    p["use_qk_l2norm_in_kernel"] = False
    p["a"], p["b"] = p["a"].contiguous(), p["b"].contiguous()
    before = p["initial_state_source"].clone()
    ref_out, ref_cache, written = fp32_reference(p, meta)
    out = fn(**p)
    torch.cuda.synchronize()
    assert count[0] == 0
    check_result(p, meta, out, ref_out, ref_cache, written, before)


def test_explicit_verify_rollback_in_fresh_sglang_process():
    env = dict(os.environ, HCU_LINEAR_ATTN_GDN_VERIFY="triton",
               HCU_ADAPTER_HOST="sglang")
    env.pop("_HCU_ADAPTER_ACTIVE_HOST", None)
    script = """
from hcu_adapter import gdn_aiter as bridge
from sglang.kernels.ops.attention.fla import fused_sigmoid_gating_recurrent as face
fn = face.fused_sigmoid_gating_delta_rule_update
assert not hasattr(fn, bridge._VERIFY_MARKER)
assert bridge._VERIFY_DISPATCH not in fn.__globals__
print("FRESH_UPSTREAM_ROLLBACK_PASSED")
"""
    out = subprocess.run([sys.executable, "-c", script], env=env,
                         capture_output=True, text=True, timeout=240)
    assert out.returncode == 0, (out.stdout[-4000:], out.stderr[-4000:])
    assert "FRESH_UPSTREAM_ROLLBACK_PASSED" in out.stdout


@pytest.mark.parametrize("n", [18,64,128])
def test_auto_first_call_inside_capture(automatic_verify_face, n):
    fn, _, count = automatic_verify_face
    p, meta = make_case(n, 4, "sglang_2d")
    p["dt_bias"] = p["dt_bias"].to(torch.bfloat16)
    p["retrieve_parent_token"] = None
    ref_out, ref_cache, written = fp32_reference(p, meta)
    before = p["initial_state_source"].clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(**p)
    graph.replay()
    torch.cuda.synchronize()
    assert count[0] == 1, "device metadata admission must work on first capture call"
    check_result(p, meta, out, ref_out, ref_cache, written, before)


def _require_large_case_memory(n):
    # One live state/snapshot pool; the full upstream result is kept on CPU.
    # Qualification must not depend on model weights fitting the same batch.
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    required = (n + 3) * 24 * 128 * 128 * 4 * 5 + (1 << 30)
    if free < required:
        pytest.skip(f"large GDN operator case needs {required} free bytes; got {free}")


def _check_sampled_fp32_rows(p, meta, output, label):
    # Independent recurrence on first/middle/last active sequences, including
    # non-identity source/snapshot mappings. Full-pool parity is checked below.
    active = [i for i, length in enumerate(meta["lengths"]) if length]
    rows = sorted({active[0], active[len(active)//2], active[-1]})
    index = torch.tensor(rows, device=p["q"].device, dtype=torch.long)
    n, steps = len(meta["lengths"]), p["cache_steps"]
    small = dict(p)
    for name in ("q", "k", "v"):
        value = p[name]
        small[name] = value.reshape(n, steps, *value.shape[-2:]).index_select(
            0, index).reshape(1, -1, *value.shape[-2:])
    for name in ("a", "b"):
        small[name] = p[name].reshape(n, steps, 24).index_select(0, index).reshape(-1, 24)
    small["intermediate_states_buffer"] = torch.empty(
        len(rows), steps, 24, 128, 128, device=p["q"].device)
    small["retrieve_parent_token"] = (None if p["retrieve_parent_token"] is None
                                     else p["retrieve_parent_token"].index_select(0, index))
    selected = dict(lengths=[steps]*len(rows), cu=[i*steps for i in range(len(rows)+1)],
                    indices=[meta["indices"][i] for i in rows],
                    scratch=list(range(len(rows))),
                    parents=[meta["parents"][i] for i in rows], used=steps*len(rows))
    ref_out, ref_cache, _ = fp32_reference(small, selected)
    actual = output.reshape(n, steps, 24, 128).index_select(0, index).reshape(1, -1, 24, 128)
    assert_values(actual, ref_out, 5e-2, label + " sampled FP32 output")
    for j, row in enumerate(rows):
        assert_values(p["intermediate_states_buffer"][meta["scratch"][row]],
                      ref_cache[j], 1e-4, label + f" sampled FP32 snapshots row={row}")


def _check_large_provider_pair(p, meta, run_aiter, run_triton, label):
    source_before = p["initial_state_source"].cpu()
    scratch = p["intermediate_states_buffer"]
    scratch.fill_(SENTINEL)
    triton_output = run_triton()
    torch.cuda.synchronize()
    _check_sampled_fp32_rows(p, meta, triton_output, label + " Triton")
    expected_output, expected_cache = triton_output.cpu(), scratch.cpu()
    assert torch.equal(p["initial_state_source"].cpu(), source_before)
    scratch.fill_(SENTINEL)
    aiter_output = run_aiter()
    torch.cuda.synchronize()
    assert_values(aiter_output.reshape(1, -1, 24, 128)[:, :meta["used"]].cpu(),
                  expected_output.reshape(1, -1, 24, 128)[:, :meta["used"]],
                  5e-2, label + " AITER/Triton output")
    # Select only written slots for numerical denominators; sentinels must not
    # hide an error in the much smaller valid states.
    slots = sorted({slot for slot, length in zip(meta["scratch"], meta["lengths"])
                    if slot >= 0 and length})
    for start in range(0, len(slots), 16):
        selected = slots[start:start+16]
        device_index = torch.tensor(selected, device=scratch.device, dtype=torch.long)
        cpu_index = torch.tensor(selected, dtype=torch.long)
        actual = scratch.index_select(0, device_index)
        expected = expected_cache.index_select(0, cpu_index).to(scratch.device)
        assert torch.isfinite(actual).all().item()
        assert_values(actual, expected, 1e-4,
                      label + f" AITER/Triton snapshots block={start//16}")
    unused = sorted(set(range(scratch.shape[0])) - set(slots))
    for slot in unused:
        assert torch.equal(scratch[slot], torch.full_like(scratch[slot], SENTINEL))
        assert torch.equal(expected_cache[slot],
                           torch.full_like(expected_cache[slot], SENTINEL))
    _check_sampled_fp32_rows(p, meta, aiter_output, label + " AITER")
    assert torch.equal(p["initial_state_source"].cpu(), source_before)
    print("LARGE_GDN_PARITY", label, "sequences", len(meta["lengths"]),
          "full_written_slots", len(slots), "untouched_slots", len(unused), flush=True)


def _change_large_graph_inputs(p, n, seed, kind):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    for name in ("q", "k", "v"):
        p[name].normal_(0, 0.3)
    for name in ("a", "b", "A_log", "dt_bias"):
        p[name].normal_(0, 0.2)
    p["initial_state_source"].normal_(0, 0.1)
    slots, steps = p["initial_state_source"].shape[0], p["cache_steps"]
    indices, destinations = rng.sample(range(slots), n), rng.sample(range(slots), n)
    lengths = [steps]*n
    parents = [[-1] + list(range(steps-1)) for _ in range(n)]
    if kind in ("tree", "padded"):
        for row in parents:
            row[2] = 0
    if kind == "padded":
        lengths[-1] = 0
        indices[-1] = destinations[-1] = -1
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    for name, value in (("initial_state_indices", indices),
                        ("intermediate_state_indices", destinations),
                        ("cu_seqlens", cu), ("retrieve_parent_token", parents)):
        if p[name] is not None:
            p[name].copy_(torch.tensor(value, device=p[name].device, dtype=p[name].dtype))
    return dict(lengths=lengths, cu=cu, indices=indices, scratch=destinations,
                parents=parents, used=cu[-1])


@pytest.mark.parametrize("n", [65,128,256,512])
@pytest.mark.parametrize("parents_enabled", [True,False])
@pytest.mark.parametrize("bias_dtype", [torch.float32,torch.bfloat16])
def test_auto_large_batch_aiter_triton_precision(
        automatic_verify_face, n, parents_enabled, bias_dtype):
    _require_large_case_memory(n)
    fn, upstream, count = automatic_verify_face
    p, meta = make_case(n, 4, "sglang_2d")
    p["dt_bias"] = p["dt_bias"].to(bias_dtype)
    if not parents_enabled:
        p["retrieve_parent_token"] = None
    _check_large_provider_pair(p, meta, lambda: fn(**p), lambda: upstream(**p),
                               f"eager N={n} tree={parents_enabled} bias={bias_dtype}")
    assert count[0] == 1, "large-batch auto must execute AITER"


@pytest.mark.parametrize("n", [128,512])
@pytest.mark.parametrize("parents_enabled", [True,False])
def test_auto_large_batch_changing_input_graph(automatic_verify_face, n, parents_enabled):
    _require_large_case_memory(n)
    fn, upstream, count = automatic_verify_face
    p, meta = make_case(n, 4, "sglang_2d")
    p["dt_bias"] = p["dt_bias"].to(torch.bfloat16)
    if not parents_enabled:
        p["retrieve_parent_token"] = None
    for _ in range(2):
        fn(**p)
        upstream(**p)
    torch.cuda.synchronize()
    aiter_graph, triton_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(aiter_graph):
        aiter_output = fn(**p)
    with torch.cuda.graph(triton_graph):
        triton_output = upstream(**p)
    def run_aiter():
        aiter_graph.replay()
        return aiter_output
    def run_triton():
        triton_graph.replay()
        return triton_output
    # Replay the original values first; capture alone is not a numerical check.
    _check_large_provider_pair(p, meta, run_aiter, run_triton, f"graph baseline N={n}")
    for seed, kind in [(31,"chain"),(37,"tree"),(43,"padded")]:
        meta = _change_large_graph_inputs(p, n, seed, kind)
        _check_large_provider_pair(p, meta, run_aiter, run_triton,
                                   f"graph N={n} tree={parents_enabled} seed={seed} {kind}")
    assert count[0] == 3, "Graph replay must not re-enter Python dispatch"
