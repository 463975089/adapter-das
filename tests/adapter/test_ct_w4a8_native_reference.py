"""Official CT input-dtype token quantization; run in a clean CPU process."""

import pytest

torch = pytest.importorskip("torch")

from tools.ct_w4a8_reference import (
    int32_native_expert, token_int8, token_int8_native,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("columns", [8, 160, 513])
def test_native_quantization_matches_official_ct(dtype, columns, request):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    from test_ct_w4a8_reference import official

    args, dynamic, fake, quantize, _, _ = official()
    rng = torch.Generator().manual_seed(825)
    value = (torch.randn(3, 17, columns, generator=rng) * 4).to(dtype)
    value[:, 0].zero_()
    value[1] *= .03125
    value[2, 1] = torch.tensor(
        [127.5, -127.5, .5, -.5, 1.5, -1.5, 2.5, -2.5],
        dtype=dtype,
    ).repeat((columns + 7) // 8)[:columns]
    original = value.clone()
    expected_scale, zp = dynamic(value, args, torch.nn.Identity())
    q, s, qdq = token_int8_native(value)
    assert q.dtype == torch.int8 and s.dtype == qdq.dtype == dtype
    assert torch.equal(s, expected_scale)
    assert torch.equal(q, quantize(value, expected_scale, zp, args, dtype=torch.int8))
    assert torch.equal(qdq, fake(value, expected_scale, zp, args))
    assert s[0, 0, 0] == torch.finfo(dtype).eps
    assert torch.equal(value, original)
    if dtype == torch.float32:
        old = token_int8(value)
        assert all(torch.equal(a, b) for a, b in zip((q, s, qdq), old))
    else:
        _, promoted_scale, _ = token_int8(value.float())
        assert not torch.equal(s.float(), promoted_scale)


@pytest.mark.parametrize("fault", ["dtype", "2d", "empty", "nan", "inf"])
def test_native_reference_refuses_invalid_inputs(fault):
    value = torch.ones(1, 2, 8)
    if fault == "dtype":
        value = value.double()
    elif fault == "2d":
        value = value[0]
    elif fault == "empty":
        value = value[:, :0]
    else:
        value[0, 0, 0] = float(fault)
    with pytest.raises(ValueError):
        token_int8_native(value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("limit", [None, .75])
def test_native_integer_expert_matches_official_codes_and_storage(dtype, limit, request):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    from test_ct_w4a8_reference import official

    args, dynamic, _, quantize, pack, _ = official()
    rng = torch.Generator().manual_seed(841)
    value = torch.randn(2, 3, 16, generator=rng).to(dtype)
    value[:, 0].zero_()

    def weight(rows, columns):
        signed = torch.randint(-8, 8, (rows, columns), generator=rng, dtype=torch.int8)
        scale = (torch.rand(rows, 1, generator=rng) * .125 + .01).bfloat16()
        return (pack(signed, 4), scale), signed

    gate, gi = weight(24, 16)
    up, ui = weight(24, 16)
    down, di = weight(16, 24)

    def independent(x, signed, scale):
        s, zp = dynamic(x, args, torch.nn.Identity())
        q = quantize(x, s, zp, args, dtype=torch.int8)
        exact = torch.matmul(q.double(), signed.double().t()).float()
        return ((exact * s.float()) * scale.float().t()).to(dtype)

    g, u = independent(value, gi, gate[1]), independent(value, ui, up[1])
    if limit is not None:
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    mid = (torch.nn.functional.silu(g.float()) * u.float()).to(dtype)
    expected = independent(mid, di, down[1])
    actual = int32_native_expert(value, gate, up, down, swiglu_limit=limit)
    assert actual.dtype == dtype and torch.equal(actual, expected)


@pytest.mark.parametrize("limit", [True, 0, -1., float("inf"), float("nan")])
def test_native_integer_expert_refuses_invalid_clamp(limit):
    value = torch.ones(1, 1, 8, dtype=torch.bfloat16)
    weight = torch.zeros(1, 1, dtype=torch.int32), torch.ones(1, 1)
    with pytest.raises(ValueError, match="swiglu_limit"):
        int32_native_expert(value, weight, weight, weight, swiglu_limit=limit)
