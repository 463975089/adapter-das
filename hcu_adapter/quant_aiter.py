"""R-09 (3): aiter-backed per-token quant family for sglang fp8_kernel.

Vendor tier behind ``HCU_QUANT_IMPL`` (auto/aiter/stock; auto = aiter on
hip when the wheel imports). Behavior restoration first, perf second
(probe tmp_fetch/probe_r09_quant.py, 2026-10-19):

* ``sgl_per_token_quant_fp8`` does not exist on HIP builds — upstream's
  rowwise wrapper ``sglang_per_token_quant_fp8`` raises NameError, and the
  whole-row shortcut inside ``sglang_per_token_group_quant_fp8`` routes
  straight into it (latent break; nothing on the bf16 serving path calls
  it — same shape as the R-08 ragged fp8_mqa_logits finding).
* the numeric tier is healthy: aiter ``per_token_group_quant_fp8`` scales
  are bit-identical to the torch golden (amax/448, row-group fp32), q
  matches RTN golden on 99.8% of elements (rest within 1 ulp), and the
  group kernel is 15.1x the torch path at 4096x8192.

Bridged entries (module-attr rebind on BOTH the kernel module and the
srt-side ``fp8_utils`` from-import copies — the double binding is the
trap):

* ``sglang_per_token_quant_fp8``        -> aiter rowwise (fixes NameError)
* ``sglang_per_token_group_quant_fp8``  -> aiter group (plain arm only)
* ``per_token_group_quant_fp8`` (hip alias of ``_..._raw``) -> aiter group
* ``_native_dynamic_per_token_quant_fp8`` / ``_native_dynamic_per_tensor_
  quant_fp8`` -> upstream's own aiter calls (exactly what
  ``SGLANG_USE_AITER=1`` would route to — no new semantics; only affects
  ``scaled_fp8_quant`` when that flag is unset)
* ``_native_static_quant_fp8`` stays native on purpose: the aiter
  ``static_per_tensor_quant`` wheel kernel does NOT clamp, so any element
  with |x/scale| > 448 becomes NaN (e4m3fn overflow), while the native
  clamps to ±448 (gate evidence 2026-10-19: 64x1024 randn*3.7 scale=0.01
  -> 21% NaN, |x|>4.48 fraction == 22.6%). Upstream's SGLANG_USE_AITER
  branch shares the NaN behavior, but this bridge promises stock parity
  or fallback, and static per-tensor activation quant is not on our
  serving path — parity wins over a marginal launch saving.

R-09 (2) seam: the group entry's ``fuse_silu_and_mul=True`` arm with a
whole-row group (``group_size == K_out``) routes to aiter
``fuse_silu_mul_per_token_quant`` (rowwise scale [T,1], 7.5x vs two-step
at 4096x2048). Narrower fused groups keep the upstream two-step path
(aiter fused is rowwise-only).

Degradation is transparent: aiter failure falls back to the saved
upstream function (torch inline for the missing rowwise op), log-once.
Exotic arms (column_major_scales / scale_tma_aligned / scale_ue8m0 /
masked_m / non-e4m3 dtypes / batched 3D inputs) always go upstream.
"""

import logging

LOG = logging.getLogger("hcu.adapter")


def install():
    import os

    from . import hcu_env, registry as _reg

    impl = hcu_env.resolve("HCU_QUANT_IMPL")[0]
    if impl == "stock":
        LOG.info("[HCU quant] HCU_QUANT_IMPL=stock -- upstream natives stay")
        return
    try:
        import torch
        from aiter import (
            dynamic_per_tensor_quant,
            dynamic_per_token_scaled_quant,
            fuse_silu_mul_per_token_quant as aiter_fused,
            per_token_group_quant_fp8 as aiter_group,
        )
        from sglang.kernels.ops.quantization import fp8_kernel as _fk
    except Exception:
        LOG.warning("[HCU quant] aiter fp8-quant tier not installed "
                    "(impl=%s)", impl, exc_info=True)
        return
    fp8_dtype = _fk.fp8_dtype

    if getattr(_fk.sglang_per_token_quant_fp8, "_hcu_quant_aiter", False):
        return  # already installed (sitecustomize + explicit install)

    sg = _fk.sglang_per_token_group_quant_fp8      # jit-shaped wrapper
    raw = _fk.per_token_group_quant_fp8            # hip alias of _raw
    nat_tok = getattr(_fk, "_native_dynamic_per_token_quant_fp8", None)
    nat_ten = getattr(_fk, "_native_dynamic_per_tensor_quant_fp8", None)

    _fell_back = []

    def _once(tag):
        if tag not in _fell_back:
            _fell_back.append(tag)
            LOG.warning("[HCU quant] aiter %s failed -- fell back to "
                        "upstream (log-once)", tag, exc_info=True)

    def _torch_rowwise(x, dtype):
        xq = torch.empty(x.shape, device=x.device, dtype=dtype)
        xs = torch.empty((x.shape[0], 1), device=x.device,
                         dtype=torch.float32)
        amax = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        s = amax / 448.0
        xs.copy_(s)
        xq.copy_((x.float() / s).clamp(-448.0, 448.0).to(dtype))
        return xq, xs

    def _aiter_group(x, group_size, eps):
        xq = torch.empty(x.shape, device=x.device, dtype=fp8_dtype)
        xs = torch.empty((x.shape[0], x.shape[-1] // group_size),
                         device=x.device, dtype=torch.float32)
        aiter_group(xq, x, xs, group_size, eps)
        return xq, xs

    def _plain(x):
        return x.dim() == 2 and x.is_contiguous() and x.shape[0] > 0

    # ---- rowwise: fixes the HIP NameError, 15x on the hot path ----------
    def _rowwise(x, dtype=None):
        dtype = dtype or fp8_dtype
        if dtype != fp8_dtype or not _plain(x):
            return _torch_rowwise(x, dtype)   # upstream semantics, inline
        try:
            xq = torch.empty(x.shape, device=x.device, dtype=fp8_dtype)
            xs = torch.empty((x.shape[0], 1), device=x.device,
                             dtype=torch.float32)
            dynamic_per_token_scaled_quant(xq, x, xs)
            return xq, xs
        except Exception:
            _once("rowwise")
            return _torch_rowwise(x, dtype)

    # ---- group: plain arm -> aiter; fused whole-row -> aiter fused ------
    def _group_sg(x, group_size, eps=1e-10, column_major_scales=False,
                  scale_tma_aligned=False, scale_ue8m0=False,
                  fuse_silu_and_mul=False, masked_m=None):
        plain = not (column_major_scales or scale_tma_aligned
                     or scale_ue8m0 or masked_m is not None)
        if plain and fuse_silu_and_mul and x.dim() == 2 \
                and x.is_contiguous() and x.shape[0] > 0 \
                and group_size == x.shape[-1] // 2:
            # R-09(2): whole-row fused silu+quant -> aiter fused kernel
            try:
                q, s = aiter_fused(x, dtype=fp8_dtype, topk=1)
                return q.view(x.shape[0], group_size), \
                    s.view(x.shape[0], 1)
            except Exception:
                _once("fused")
        elif plain and not fuse_silu_and_mul \
                and group_size == x.shape[-1] and x.dim() == 2:
            return _rowwise(x)  # upstream whole-row shortcut target
        elif plain and not fuse_silu_and_mul \
                and _plain(x) and x.shape[-1] % group_size == 0:
            try:
                return _aiter_group(x, group_size, eps)
            except Exception:
                _once("group")
        return sg(x, group_size, eps=eps,
                  column_major_scales=column_major_scales,
                  scale_tma_aligned=scale_tma_aligned,
                  scale_ue8m0=scale_ue8m0,
                  fuse_silu_and_mul=fuse_silu_and_mul, masked_m=masked_m)

    def _group_raw(x, group_size, eps=1e-10, dtype=None,
                   column_major_scales=False, scale_tma_aligned=False,
                   scale_ue8m0=False):
        dtype = dtype or fp8_dtype
        plain = not (column_major_scales or scale_tma_aligned
                     or scale_ue8m0)
        if plain and dtype == fp8_dtype and _plain(x) \
                and x.shape[-1] % group_size == 0:
            if group_size == x.shape[-1]:
                return _rowwise(x)
            try:
                return _aiter_group(x, group_size, eps)
            except Exception:
                _once("group")
        return raw(x, group_size, eps=eps, dtype=dtype,
                   column_major_scales=column_major_scales,
                   scale_tma_aligned=scale_tma_aligned,
                   scale_ue8m0=scale_ue8m0)

    # ---- scaled_fp8_quant natives: upstream's own aiter calls -----------
    def _nat(nat, aiter_fn, tag):
        def _w(output, input, scale):
            try:
                aiter_fn(output, input, scale)
            except Exception:
                _once(tag)
                nat(output, input, scale)
        return _w

    targets = {
        "sglang_per_token_quant_fp8": _rowwise,
        "sglang_per_token_group_quant_fp8": _group_sg,
        "per_token_group_quant_fp8": _group_raw,
    }
    if nat_tok is not None:
        targets["_native_dynamic_per_token_quant_fp8"] = \
            _nat(nat_tok, dynamic_per_token_scaled_quant, "scaled-row")
    if nat_ten is not None:
        targets["_native_dynamic_per_tensor_quant_fp8"] = \
            _nat(nat_ten, dynamic_per_tensor_quant, "scaled-tensor")
    # _native_static_quant_fp8 deliberately NOT rebound: aiter's static
    # kernel overflows to NaN where the native clamps (see module header).

    try:
        from sglang.srt.layers.quantization import fp8_utils as _fu
    except Exception:
        _fu = None
    for name, fn in targets.items():
        fn._hcu_quant_aiter = True   # idempotency marker
        setattr(_fk, name, fn)
        if _fu is not None and hasattr(_fu, name):
            setattr(_fu, name, fn)   # srt-side from-import copies

    _reg.register(
        "per_token_quant", _rowwise,
        target="sglang.kernels.ops.quantization.fp8_kernel."
               "sglang_per_token_quant_fp8",
        mechanism=_reg.MECH_REBIND,
        detail="aiter tier (HCU_QUANT_IMPL=%s): rowwise NameError fix + "
               "group plain arm + scaled_fp8_quant natives + whole-row "
               "fused silu+quant (R-09 2/3)" % impl,
    )
    LOG.warning("[HCU quant] aiter fp8-quant tier installed "
                "(HCU_QUANT_IMPL=%s): rowwise/group/scaled-natives rebind "
                "+ whole-row fused silu+quant arm", impl)
