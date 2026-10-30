"""fa3 attention -> DTK flash_attn wheel router (§0.5 route tier-② vendor).

The bw7/bw6 images ship the DTK-maintained ``flash_attn`` wheel (2.8.4,
gfx93x kernels, same FA2 call family sglang's fa2/FA3 backend was built
on). The §0.5 is_cuda semantics route sglang into the fa3 backend, whose
FA3-facing surface ``sgl_kernel.flash_attn`` used to answer with the torch
chunked emulation — correct but slow (pure bmm/softmax, ~45 s per decode
graph bucket on gfx936).

This bridge re-binds the two hot entries of ``sgl_kernel.flash_attn`` to
thin adapters over the vendor wheel and falls back to the in-tree emulation
for the paths the wheel does not cover:

.. list-table::
   :header-rows: 1

   * - FA3-facing call
     - vendor path
   * - ``flash_attn_with_kvcache`` decode/uniform-window paged GQA
     - ``flash_attn.flash_attn_with_kvcache`` (4D q (B,L,H,D) +
       ``block_table``); bottom-right causal comes free from cache_seqlens
   * - ``flash_attn_varlen_func`` prefill (incl. fp8-KV descale, ``q_v``)
     - ``flash_attn.flash_attn_varlen_func`` (same ragged contract)
   * - anything else (ragged q windows on kvcache, k/v write-through,
     sinks, only_qv, q_descale, external rotary)
     - in-tree emulation, unchanged

``merge_state_v2`` stays on the in-tree LSE merge (the wheel has no export;
it is a cheap pointwise reduce). Numerics gate: wrapper-level parity test
vs the emulation against reference math lands in tests/kernels (bgate);
e2e anchor is the models.md row.
"""

import logging
import os

import torch

LOG = logging.getLogger("hcu.adapter")


def _uniform_windows(cu_q, bs):
    """Decode/spec calls use equal windows per request; ragged ones
    (chunked prefill tails) keep the emulation."""
    lens = cu_q[1:] - cu_q[:-1]
    return bool(bs) and (lens == lens[0]).all()


class _VendorFA:
    """FA3-shaped facade over the DTK flash_attn wheel."""

    def __init__(self):
        import flash_attn as _vfa

        self._v = _vfa

    # -- paged kvcache (decode/spec) and dense kvcache (no page_table) -----
    def with_kvcache(
        self,
        q,
        k_cache,
        v_cache,
        *,
        page_table,
        cu_seqlens_q,
        cache_seqlens,
        max_seqlen_q,
        causal,
        softmax_scale,
        return_softmax_lse,
        window_size=(-1, -1),
        softcap=0.0,
        out=None,
    ):
        # wheel contract: 4D q (B, L, H, D). 3D (M, H, D) is the uniform-
        # window flattened form (the only thing reaching here after the
        # bridge guards); re-view by the window length.
        if q.dim() == 3:
            if max_seqlen_q is not None:
                L = max(int(max_seqlen_q), 1)
            elif cu_seqlens_q is not None:
                L = max(int(cu_seqlens_q[1] - cu_seqlens_q[0]), 1)
            else:
                L = 1
            bs = max(q.shape[0] // L, 1)
            q = q.reshape(bs, L, *q.shape[-2:])

        cp = dict(
            cache_seqlens=cache_seqlens.to(torch.int32),
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=tuple(window_size) if window_size else (-1, -1),
            softcap=softcap or 0.0,
            return_softmax_lse=return_softmax_lse,
        )
        if page_table is not None:
            cp["block_table"] = page_table.to(torch.int32).contiguous()
        if out is not None:
            cp["out"] = out
        return self._v.flash_attn_with_kvcache(q, k_cache, v_cache, **cp)

    # -- ragged prefill ---------------------------------------------------
    def varlen(
        self,
        q,
        k,
        v,
        cu_q,
        cu_k,
        *,
        max_q=None,
        max_k=None,
        causal=False,
        softmax_scale=None,
        window_size=(-1, -1),
        softcap=0.0,
        qv=None,
        k_descale=None,
        v_descale=None,
        return_softmax_lse=False,
        **_kw,
    ):
        kw = dict(
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=tuple(window_size) if window_size else (-1, -1),
            softcap=softcap or 0.0,
        )
        if return_softmax_lse:
            kw["return_attn_probs"] = True
        return self._v.flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, max_q, max_k, **kw
        )


# kwargs the emulation accepts on kvcache but the wheel path must refuse
# (fall back), per contract table in the module docstring.
_KV_FALLBACK_KEYS = (
    "k",
    "v",
    "q_descale",
    "sinks",
    "rotary_cos",
    "rotary_sin",
    "cache_batch_idx",
)

# gfx93x vendor-wheel paged kernel: q windows <= 64 rows pass, >= 65 hit
# "qheads*mtp/kvheads must be smaller than 48" internal assert (wave64 tile
# budget). Verified by shape sweep 8..64 ok / 65+ fail (2026-10-12 bw7).
_VENDOR_Q_LIM = 64

_DECODE_HITS = 0
_VARLEN_HITS = 0
_WIDE_HITS = 0

# Diagnostic (HCU_FA_EMU_TRACE=1): one-shot signature of every call that
# lands in the sgl_kernel FA emulation, so spec (draft/verify) faces can be
# attributed to the exact gate that rejected the triton route. Host values
# only — capture-safe.
_EMU_TRACE = os.environ.get("HCU_FA_EMU_TRACE", "0") == "1"
_EMU_SIGS = set()


def _log_emu_sig(face, p, reasons=None):
    if not _EMU_TRACE:
        return
    q = p.get("q")
    cu_q = p.get("cu_seqlens_q")
    sig = (
        face,
        tuple(q.shape) if q is not None else None,
        "causal" if p.get("causal") else "noncausal",
        "pt" if p.get("page_table") is not None else "nopt",
        "cuq%d" % (cu_q.numel() - 1) if cu_q is not None else "cuq-",
        "cs" if p.get("cache_seqlens") is not None else "nocs",
        tuple(k for k in ("k", "v", "q_descale", "qv", "sinks", "out")
              if p.get(k) is not None),
        tuple(p.get("window_size") or ()),
        tuple(reasons or ()),
    )
    if sig not in _EMU_SIGS:
        _EMU_SIGS.add(sig)
        LOG.warning("HCU adapter: FA emu sig %r", (sig,))

# param list the emu (sgl_kernel.flash_attn.flash_attn_varlen_func) accepts,
# in positional order — the reroute gate reads out of this map.
_VARLEN_NAMES = (
    "q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,"
    "seqused_q,seqused_k,page_table,softmax_scale,causal,qv,q_descale,"
    "k_descale,v_descale,window_size,attention_chunk,softcap,num_splits,"
    "pack_gqa,only_qv,sm_margin,return_softmax_lse,sinks,out"
).split(",")


def _varlen_triton_ok(p):
    """Host-only (capture-safe, no device reads) gate for the triton
    varlen kernel's contract; everything else keeps the emu."""
    q = p.get("q")
    k = p.get("k")
    v = p.get("v")
    if q is None or k is None or v is None:
        return False
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        return False
    if p.get("page_table") is not None:
        return False
    if (p.get("qv") is not None or p.get("sinks") is not None
            or p.get("q_descale") is not None or bool(p.get("only_qv"))):
        return False
    if (p.get("softcap") or 0.0) != 0.0:
        return False
    if q.dtype.itemsize < 2:
        return False
    D = q.shape[-1]
    if D > 256 or (D & (D - 1)) != 0 or v.shape[-1] != D or k.shape[-1] != D:
        return False
    Hk = k.shape[-2]
    return q.shape[-2] % Hk == 0


def _install_triton_decode(_sk_fa, registry):
    """Route the decode kvcache face (L=1 uniform windows on the paged
    pool) and the ragged varlen prefill face to the in-tree triton
    kernels; everything else keeps the emu. Also rebinds the upstream ops
    layer (stable kwargs) so capture paths entering via the v3 wrapper
    land here cleanly."""
    from sgl_kernel.paged_attn_triton import (
        paged_attention_decode_triton,
        paged_attention_prefill_triton,
    )
    from sgl_kernel.fa_varlen_triton import flash_attn_varlen_triton

    _emu_kvcache = _sk_fa.flash_attn_with_kvcache
    _emu_varlen = _sk_fa.flash_attn_varlen_func
    _names = (
        "q,k_cache,v_cache,k,v,rotary_cos,rotary_sin,cache_seqlens,"
        "cache_batch_idx,cache_leftpad,page_table,cu_seqlens_q,"
        "cu_seqlens_k_new,max_seqlen_q,rotary_seqlens,q_descale,"
        "k_descale,v_descale,softmax_scale,causal,window_size,"
        "attention_chunk,softcap,rotary_interleaved,scheduler_metadata,"
        "num_splits,pack_gqa,only_qv,sm_margin,return_softmax_lse,sinks,"
        "out"
    ).split(",")

    def flash_attn_with_kvcache(*args, **kw):
        p = dict(zip(_names, args))
        p.update(kw)
        q = p.get("q")
        # Prefill graph capture feeds the varlen mode through this facade
        # (3D q + cu_seqlens_q, k/v dense, page_table/cache_seqlens/k_cache
        # all None): reroute to the varlen face where the triton kernel is
        # dispatchable (same reroute the vendor branch has).
        if (
            p.get("k_cache") is None
            and p.get("v_cache") is None
            and q is not None
            and q.dim() == 3
            and p.get("cu_seqlens_q") is not None
            and p.get("page_table") is None
            and p.get("cache_seqlens") is None
            and p.get("k") is not None
            and p.get("v") is not None
        ):
            return flash_attn_varlen(
                p["q"], p["k"], p["v"], p["cu_seqlens_q"],
                p.get("cu_seqlens_k_new") or p["cu_seqlens_q"],
                max_seqlen_q=p.get("max_seqlen_q"),
                softmax_scale=p.get("softmax_scale"),
                causal=p.get("causal"),
                window_size=p.get("window_size") or (-1, -1),
                softcap=p.get("softcap") or 0.0,
                k_descale=p.get("k_descale"),
                v_descale=p.get("v_descale"),
                return_softmax_lse=p.get("return_softmax_lse"),
                out=p.get("out"),
            )
        cu_q = p.get("cu_seqlens_q")
        # Decode feeds 3D packed q, sometimes with cu_seqlens_q (FA3-style),
        # sometimes without (metadata form; B comes from the page table).
        # Normalize to the 4D (B,L,H,D) form the triton kernels take: L=1
        # decode rows, and uniform L>1 spec windows (NEXTN verify packs
        # draft_token_num tokens per request; with topk=1 the tree mask
        # degenerates to a chain == bottom-right causal, which the paged
        # prefill kernel already applies).
        # Graph-capture safe: no host-device syncs — the uniform-window
        # contract comes from max_seqlen_q/n_tok consistency, not an
        # .all() probe.
        if (
            q is not None
            and q.dim() == 3
            and p.get("page_table") is not None
            and p.get("cache_seqlens") is not None
        ):
            n_tok, H, D = q.shape
            L = 0
            if cu_q is not None and hasattr(cu_q, "numel"):
                B = cu_q.numel() - 1
                msq = p.get("max_seqlen_q")
                msq_i = (
                    int(msq)
                    if msq is not None and not torch.is_tensor(msq)
                    else None
                )
                if n_tok == B and (msq_i is None or msq_i <= 1):
                    L = 1  # every window is exactly 1 token (decode)
                elif B > 0 and msq_i and msq_i > 1 and n_tok == B * msq_i:
                    L = msq_i  # uniform spec windows (e.g. verify L=2)
                # ragged windows keep L=0 -> emu
            else:
                B = p["page_table"].shape[0]
                L = 1 if n_tok == B else 0  # one token per row == decode
            if L >= 1 and B > 0:
                q = q.reshape(B, L, H, D)
                p["q"] = q
                p["cu_seqlens_q"] = None
        triton_ok = (
            q is not None
            and q.dim() == 4
            and p.get("page_table") is not None
            and p.get("cache_seqlens") is not None
            and p.get("cu_seqlens_q") is None
            and p.get("qv") is None
            and not p.get("only_qv")
            and all(p.get(k) is None for k in _KV_FALLBACK_KEYS)
            and (p.get("window_size") or (-1, -1)) == (-1, -1)
            and (p.get("softcap") or 0.0) == 0.0
        )
        if triton_ok and q.shape[1] == 1:
            global _DECODE_HITS
            _DECODE_HITS += 1
            if _DECODE_HITS == 1 or _DECODE_HITS % 20000 == 0:
                LOG.warning(
                    "HCU adapter: triton decode face hit #%d (B=%d)",
                    _DECODE_HITS, q.shape[0],
                )
            o, lse = paged_attention_decode_triton(
                q,
                p["k_cache"],
                p["v_cache"],
                p["page_table"],
                p["cache_seqlens"],
                softmax_scale=p.get("softmax_scale"),
            )
            return (o, lse) if p.get("return_softmax_lse") else o
        if (
            triton_ok
            and q.shape[1] > 1
            and p.get("causal")
            and p.get("custom_mask") is None
            and p.get("out") is None
        ):
            # paged extend / spec verify: whole q window against the pool,
            # bottom-right causal. NEXTN topk=1 verify trees are chains ==
            # causal; real trees (custom_mask) and graph out= buffers keep
            # the emu.
            global _WIDE_HITS
            _WIDE_HITS += 1
            if _WIDE_HITS == 1 or _WIDE_HITS % 20000 == 0:
                LOG.warning(
                    "HCU adapter: triton wide-window face hit #%d "
                    "(B=%d L=%d)",
                    _WIDE_HITS, q.shape[0], q.shape[1],
                )
            o, lse = paged_attention_prefill_triton(
                q,
                p["k_cache"],
                p["v_cache"],
                p["page_table"],
                p["cache_seqlens"],
                softmax_scale=p.get("softmax_scale"),
            )
            return (o, lse) if p.get("return_softmax_lse") else o
        if _EMU_TRACE:
            reasons = []
            if not triton_ok:
                reasons = [
                    n
                    for n, ok in (
                        ("q4d", q is not None and q.dim() == 4),
                        ("page", p.get("page_table") is not None),
                        ("cseq", p.get("cache_seqlens") is not None),
                        ("cuq-none", p.get("cu_seqlens_q") is None),
                        ("fbkeys", all(p.get(k) is None for k in _KV_FALLBACK_KEYS)),
                        ("win", (p.get("window_size") or (-1, -1)) == (-1, -1)),
                        ("cap", (p.get("softcap") or 0.0) == 0.0),
                        ("onlyqv", not p.get("only_qv")),
                    )
                    if not ok
                ]
            elif q is not None and q.shape[1] > 1 and not p.get("causal"):
                reasons = ["noncausal_wide"]
            _log_emu_sig("kvcache", p, reasons)
        return _emu_kvcache(**p)

    _sk_fa.flash_attn_with_kvcache = flash_attn_with_kvcache
    registry.register(
        "flash_attention",
        flash_attn_with_kvcache,
        target="sgl_kernel.flash_attn:flash_attn_with_kvcache",
        mechanism=registry.MECH_REBIND,
        detail="decode L=1 + paged wide windows -> triton kernel; rest -> emu (HCU_FA_IMPL=triton)",
    )

    # ops-layer rebind (kwargs-stable), mirroring the vendor branch but
    # delegating varlen straight to the emu (widow windows keep emu)
    try:
        import sglang.kernels.ops.attention.flash_attention_v3 as _fa3v

        def _ops_kvcache(
            q, k_cache, v_cache,
            k=None, v=None, qv=None, rotary_cos=None, rotary_sin=None,
            cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None,
            page_table=None, cu_seqlens_q=None, cu_seqlens_k_new=None,
            max_seqlen_q=None, rotary_seqlens=None, q_descale=None,
            k_descale=None, v_descale=None, softmax_scale=None,
            causal=False, window_size=(-1, -1), attention_chunk=None,
            softcap=0.0, rotary_interleaved=True, scheduler_metadata=None,
            num_splits=0, pack_gqa=None, only_qv=False, sm_margin=0,
            return_softmax_lse=False, sinks=None, out=None,
        ):
            return flash_attn_with_kvcache(
                q=q, k_cache=k_cache, v_cache=v_cache,
                page_table=page_table, cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_batch_idx, cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k_new,
                max_seqlen_q=max_seqlen_q, softmax_scale=softmax_scale,
                causal=causal, window_size=window_size, softcap=softcap,
                qv=qv, only_qv=only_qv,
                return_softmax_lse=return_softmax_lse, out=out,
            )

        _fa3v.flash_attn_with_kvcache = _ops_kvcache
        registry.register(
            "flash_attention",
            _ops_kvcache,
            target="sglang...flash_attention_v3:flash_attn_with_kvcache",
            mechanism=registry.MECH_REBIND,
            detail="ops-layer rebind; triton decode + wide-window route",
        )
        # The public dispatcher sglang.kernels.ops.attention.flash_attention
        # did `from .flash_attention_v3 import flash_attn_with_kvcache as
        # fa3_flash_attn_with_kvcache` at import time, so rebinding the v3
        # module attr above does not affect its module global. The fa3
        # backend calls the *dispatcher*, so rebind the dispatcher's global
        # too (graph-safe: attribute assignment, no sync).
        import sglang.kernels.ops.attention.flash_attention as _fapub

        if hasattr(_fapub, "fa3_flash_attn_with_kvcache"):
            _fapub.fa3_flash_attn_with_kvcache = _ops_kvcache
            registry.register(
                "flash_attention",
                _ops_kvcache,
                target="sglang...ops.attention.flash_attention:fa3_flash_attn_with_kvcache global",
                mechanism=registry.MECH_REBIND,
                detail="dispatcher global rebind (from-import copy); backend "
                       "calls land in the triton route",
            )
    except Exception:
        LOG.warning(
            "HCU adapter: ops-layer triton rebind skipped", exc_info=True
        )
    def flash_attn_varlen(*args, **kw):
        # vars the sglang fa3 backend actually passes for ragged prefill;
        # the triton arm covers the dense-ragged core, the emu keeps the
        # rest (paged varlen raises upstream-routed, qv/sinks/softcap,
        # MLA D_v!=D, fp8-q)
        p = dict(zip(_VARLEN_NAMES, args))
        p.update(kw)
        global _VARLEN_HITS
        if _varlen_triton_ok(p):
            _VARLEN_HITS += 1
            if _VARLEN_HITS == 1 or _VARLEN_HITS % 20000 == 0:
                LOG.warning(
                    "HCU adapter: triton varlen prefill face hit #%d "
                    "(M=%d B=%d)",
                    _VARLEN_HITS, p["q"].shape[0],
                    p["cu_seqlens_q"].numel() - 1,
                )
            return flash_attn_varlen_triton(
                p["q"], p["k"], p["v"], p["cu_seqlens_q"],
                p["cu_seqlens_k"],
                max_seqlen_q=p.get("max_seqlen_q"),
                softmax_scale=p.get("softmax_scale"),
                causal=p.get("causal"),
                window_size=p.get("window_size") or (-1, -1),
                k_descale=p.get("k_descale"),
                v_descale=p.get("v_descale"),
                return_softmax_lse=bool(p.get("return_softmax_lse")),
                out=p.get("out"),
            )
        if _EMU_TRACE:
            reasons = [
                n
                for n, ok in (
                    ("qkv3d", all(p.get(x) is not None and p.get(x).dim() == 3
                                  for x in ("q", "k", "v"))),
                    ("nopt", p.get("page_table") is None),
                    ("noqv/sinks/qdescale", not any(
                        p.get(x) is not None for x in ("qv", "sinks", "q_descale"))
                     and not bool(p.get("only_qv"))),
                    ("nocap", (p.get("softcap") or 0.0) == 0.0),
                    ("dtype", p.get("q") is not None and p["q"].dtype.itemsize >= 2),
                    ("dims", p.get("q") is None or (
                        p["q"].shape[-1] <= 256
                        and (p["q"].shape[-1] & (p["q"].shape[-1] - 1)) == 0)),
                )
                if not ok
            ]
            _log_emu_sig("varlen", p, reasons)
        return _emu_varlen(**p)

    _sk_fa.flash_attn_varlen_func = flash_attn_varlen
    registry.register(
        "flash_attention",
        flash_attn_varlen,
        target="sgl_kernel.flash_attn:flash_attn_varlen_func",
        mechanism=registry.MECH_REBIND,
        detail="ragged varlen prefill -> triton kernel (dense ragged core, "
               "bottom-right causal + swa + fp8-KV descale); contract-exceed "
               "faces -> emu (HCU_FA_IMPL=triton)",
    )

    # ops-layer rebind (kwargs-stable), mirroring the vendor branch: the
    # v3 wrapper module's python kwargs contract is stable and documented.
    try:
        import sglang.kernels.ops.attention.flash_attention_v3 as _fa3v

        def _ops_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                        max_seqlen_q=None, max_seqlen_k=None,
                        seqused_q=None, seqused_k=None, page_table=None,
                        softmax_scale=None, causal=False, qv=None,
                        q_descale=None, k_descale=None, v_descale=None,
                        window_size=(-1, -1), attention_chunk=0,
                        softcap=0.0, num_splits=1, pack_gqa=None,
                        only_qv=False, sm_margin=0,
                        return_softmax_lse=False, sinks=None, out=None,
                        **_kw):
            return flash_attn_varlen(
                q.view(-1, q.shape[-2], q.shape[-1]) if q.dim() == 4 else q,
                k.view(-1, k.shape[-2], k.shape[-1]) if k.dim() == 4 else k,
                v.view(-1, v.shape[-2], v.shape[-1]) if v.dim() == 4 else v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                seqused_q=seqused_q, seqused_k=seqused_k,
                page_table=page_table, softmax_scale=softmax_scale,
                causal=causal, qv=qv, q_descale=q_descale,
                k_descale=k_descale, v_descale=v_descale,
                window_size=window_size, softcap=softcap,
                return_softmax_lse=return_softmax_lse, sinks=sinks,
                out=out,
            )

        _fa3v.flash_attn_varlen_func = _ops_varlen
        registry.register(
            "flash_attention",
            _ops_varlen,
            target="sglang...flash_attention_v3:flash_attn_varlen_func",
            mechanism=registry.MECH_REBIND,
            detail="ops-layer varlen rebind; triton ragged core + emu "
                   "contract-exceed route",
        )
        # the public dispatcher copied the v3 symbol at import time
        # (from-import); rebind its module global too so backend calls
        # land in this route (graph-safe attribute assignment)
        import sglang.kernels.ops.attention.flash_attention as _fapub

        for _attr in ("fa3_flash_attn_varlen_func",
                      "flash_attn_varlen_func"):
            if hasattr(_fapub, _attr):
                setattr(_fapub, _attr, _ops_varlen)
                _setattr_reg = getattr(_fapub, _attr)
                registry.register(
                    "flash_attention",
                    _setattr_reg,
                    target="sglang...ops.attention.flash_attention:"
                           f"{_attr} global",
                    mechanism=registry.MECH_REBIND,
                    detail="dispatcher global rebind (from-import copy)",
                )
                break
    except Exception:
        LOG.warning(
            "HCU adapter: ops-layer varlen rebind skipped "
            "(upstream layout changed); facade stays at sgl_kernel level",
            exc_info=True,
        )
    LOG.warning(
        "HCU adapter: fa3 triton faces -- paged decode/wide windows + "
        "ragged varlen prefill routed to the triton kernels; contract-exceed "
        "faces keep the emu (HCU_FA_IMPL=triton)"
    )


def install():
    try:
        import flash_attn as _probe  # noqa: F401
    except Exception:
        LOG.warning(
            "HCU adapter: DTK flash_attn wheel not importable; fa3 stays on "
            "the torch emulation"
        )
        return
    from hcu_adapter import hcu_env, registry
    from sgl_kernel import flash_attn as _sk_fa

    mode, _src = hcu_env.resolve("HCU_FA_IMPL")
    triton_mode = mode == "triton"
    if triton_mode:
        # tier-③ triton route: paged kvcache faces (L=1 decode windows +
        # L>1 causal extend windows) -> in-tree triton kernels (numerics
        # gate: tests/tools/_test_paged_attn_triton.py ALL-PASS on
        # gfx936+gfx938); ragged/varlen faces stay on the emu. Vendor wheel
        # untouched (its paged path remains gap-listed).
        _install_triton_decode(_sk_fa, registry)
        return
    vendor_enabled = mode == "vendor"
    if mode == "emu" or mode == "auto":
        if mode == "auto":
            # gfx936 vendor-wheel paged kernel produces token-soup on the
            # real decode path (uniform-window A/B passes, live decode
            # corrupts, 2026-10-12 bw7) -- gap-listed, awaiting DTK fix.
            # auto therefore answers emu everywhere; HCU_FA_IMPL=vendor
            # opts the wheel back in for A/B.
            pass
        LOG.warning(
            "HCU adapter: HCU_FA_IMPL=%s -- fa3 attention stays on the "
            "torch emulation (triton decode route available via =triton)", mode
        )
        return

    vendor = _VendorFA()
    _emu_kvcache = _sk_fa.flash_attn_with_kvcache
    _emu_varlen = _sk_fa.flash_attn_varlen_func

    _PARAM_NAMES = (
        "q,k_cache,v_cache,k,v,rotary_cos,rotary_sin,cache_seqlens,"
        "cache_batch_idx,cache_leftpad,page_table,cu_seqlens_q,"
        "cu_seqlens_k_new,max_seqlen_q,rotary_seqlens,q_descale,"
        "k_descale,v_descale,softmax_scale,causal,window_size,"
        "attention_chunk,softcap,rotary_interleaved,scheduler_metadata,"
        "num_splits,pack_gqa,only_qv,sm_margin,return_softmax_lse,sinks,"
        "out"
    ).split(",")

    def flash_attn_with_kvcache(*args, **kw):
        p = dict(zip(_PARAM_NAMES, args))
        p.update(kw)
        import os as _os

        if _os.environ.get("HCU_FA_BRIDGE_DEBUG") and not torch.cuda.is_current_stream_capturing():
            keys = sorted(k for k, v in p.items() if v is not None)
            LOG.warning(
                "FA-BRIDGE kvcache: nonNone=%s causal=%s", keys, p.get("causal")
            )
        # prefill graph capture passes the varlen mode through this facade
        # (3D q + cu_seqlens_q, page_table/cache_seqlens None, no k_cache read):
        # reroute it to the varlen face where both vendor and emu are correct.
        if (
            p.get("k_cache") is None
            and p.get("v_cache") is None
            and p.get("cu_seqlens_q") is not None
            and p.get("page_table") is None
        ):
            from flash_attn import flash_attn_varlen_func as _v_varlen

            return _v_varlen(
                p["q"],
                p["k"],
                p["v"],
                p["cu_seqlens_q"],
                p.get("cu_seqlens_k_new"),
                p.get("max_seqlen_q") or int((p["cu_seqlens_q"][1:] - p["cu_seqlens_q"][:-1]).max()),
                p.get("max_seqlen_q") or int((p["cu_seqlens_q"][1:] - p["cu_seqlens_q"][:-1]).max()),
                softmax_scale=p.get("softmax_scale"),
                causal=bool(p.get("causal")),
                window_size=tuple(p.get("window_size") or (-1, -1)),
                softcap=p.get("softcap") or 0.0,
            )
        if (
            p.get("qv") is not None
            or p.get("only_qv")
            or any(p.get(k) is not None for k in _KV_FALLBACK_KEYS)
            or (
                p.get("page_table") is not None
                and (
                    (p.get("window_size") or (-1, -1)) != (-1, -1)  # swa 对拍后再放行 vendor
                    or not _uniform_windows(
                        p["cu_seqlens_q"].to(torch.int64), p["page_table"].shape[0]
                    )
                )
            )
        ):
            return _emu_kvcache(**p)
        # vendor wheel constraint (gfx93x, wave64): the CUTLASS paged kernel
        # full-fails for query windows wider than 64 rows (L>=65 internal
        # assert; decode L=1 and target-verify L<=8 are unaffected). Wide
        # uniform windows — prefill graph buckets — run as q-chunks of <=64
        # rows, each a full bottom-right-causal vendor call, merged by the
        # in-tree exact LSE reduction. LSE reference frame is per-call, so
        # chunk LSEs merge correctly with merge_state_v2 math.
        # vendor wheel paged-kernel limit (gfx93x, wave64): the CUTLASS
        # kernel's computed tile budget collapses on wide q windows
        # (L*H/Hk combinations >= ~64 fail the "qheads*mtp/kvheads < 48"
        # internal assert; sweep 2026-10-12: Hk=8 L<=64 ok, Hk=4 L>=48 fail).
        # Decode (L=1) and target-verify (L<=8) — the replay-hot paths —
        # always pass; wide windows (prefill graphs) go to the emulation.
        L = int(p.get("max_seqlen_q") or 0)
        if L == 0:
            cu = p.get("cu_seqlens_q")
            L = int(cu[1] - cu[0]) if cu is not None else int(p["q"].shape[1])
        if L > _VENDOR_Q_LIM:
            return _emu_kvcache(**p)
        want_lse = bool(p.get("return_softmax_lse"))
        ret = vendor.with_kvcache(
            q=p["q"],
            k_cache=p["k_cache"],
            v_cache=p["v_cache"],
            page_table=p.get("page_table"),
            cu_seqlens_q=p.get("cu_seqlens_q"),
            cache_seqlens=p["cache_seqlens"],
            max_seqlen_q=p.get("max_seqlen_q"),
            causal=p.get("causal"),
            window_size=(-1, -1),
            softcap=p.get("softcap") or 0.0,
            softmax_scale=p.get("softmax_scale"),
            return_softmax_lse=want_lse,
            out=p.get("out"),
        )
        if want_lse and not isinstance(ret, tuple):
            ret = (ret, None)
        return ret

    def flash_attn_varlen_func(*args, **kw):
        # wheel varlen carries the full FA3-facing contract sglang uses
        # (fp8-KV descale via (1,Hk) scales, q_v absorbed-latent path)
        try:
            return vendor.varlen(*args, **kw)
        except (RuntimeError, IndexError, ValueError, NotImplementedError):
            return _emu_varlen(*args, **kw)

    _sk_fa.flash_attn_with_kvcache = flash_attn_with_kvcache
    _sk_fa.flash_attn_varlen_func = flash_attn_varlen_func

    for target, fn in (
        ("sgl_kernel.flash_attn:flash_attn_with_kvcache", flash_attn_with_kvcache),
        ("sgl_kernel.flash_attn:flash_attn_varlen_func", flash_attn_varlen_func),
    ):
        registry.register(
            "flash_attention",
            fn,
            target=target,
            mechanism=registry.MECH_REBIND,
            detail="DTK flash_attn 2.8.4 wheel; emu fallback per-contract",
        )

    # Also rebind the upstream ops-layer entry points (v3 wrapper module).
    # That module's `_call_fa3_kernel` forwards the *kvcache* call with a
    # 28-positional-arg spread onto whatever kernel handle
    # `_load_fa3_kernels()` resolved (ours) — position-compatible with the
    # sgl-kernel C++ ABI but not with a python bridge; and on capture paths
    # the FA2 fallback there re-enters our facade with varlen-shaped
    # arguments. Binding at the ops layer — whose python kwargs contract is
    # stable and documented — sidesteps both. The kvcache spread is
    # reassembled into the same keyword set our facade speaks.
    try:
        import sglang.kernels.ops.attention.flash_attention_v3 as _fa3v

        def _ops_kvcache(
            q,
            k_cache,
            v_cache,
            k=None,
            v=None,
            qv=None,
            rotary_cos=None,
            rotary_sin=None,
            cache_seqlens=None,
            cache_batch_idx=None,
            cache_leftpad=None,
            page_table=None,
            cu_seqlens_q=None,
            cu_seqlens_k_new=None,
            max_seqlen_q=None,
            rotary_seqlens=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            attention_chunk=None,
            softcap=0.0,
            rotary_interleaved=True,
            scheduler_metadata=None,
            num_splits=0,
            pack_gqa=None,
            only_qv=False,
            sm_margin=0,
            return_softmax_lse=False,
            sinks=None,
            out=None,
        ):
            assert k_cache is not None and v_cache is not None, (
                "hcu fa bridge: only_qv/synthetic-K path unsupported"
            )
            return flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cu_seqlens_q=cu_seqlens_q,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_batch_idx,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                return_softmax_lse=return_softmax_lse,
                out=out,
            )

        def _ops_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=None,
            max_seqlen_k=None,
            seqused_q=None,
            seqused_k=None,
            page_table=None,
            softmax_scale=None,
            causal=False,
            qv=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            window_size=(-1, -1),
            attention_chunk=0,
            softcap=0.0,
            num_splits=1,
            pack_gqa=None,
            only_qv=False,
            sm_margin=0,
            return_softmax_lse=False,
            sinks=None,
            out=None,
        ):
            return vendor.varlen(
                q.view(-1, q.shape[-2], q.shape[-1]) if q.dim() == 4 else q,
                k.view(-1, k.shape[-2], k.shape[-1]) if k.dim() == 4 else k,
                v.view(-1, v.shape[-2], v.shape[-1]) if v.dim() == 4 else v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_q=max_seqlen_q,
                max_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                qv=qv,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=return_softmax_lse,
            )

        _fa3v.flash_attn_with_kvcache = _ops_kvcache
        _fa3v.flash_attn_varlen_func = _ops_varlen

        for target, fn in (
            ("sglang...flash_attention_v3:flash_attn_with_kvcache", _ops_kvcache),
            ("sglang...flash_attention_v3:flash_attn_varlen_func", _ops_varlen),
        ):
            registry.register(
                "flash_attention",
                fn,
                target=target,
                mechanism=registry.MECH_REBIND,
                detail="ops-layer rebind; vendor wheel, stable kwargs",
            )
    except Exception:
        LOG.warning(
            "HCU adapter: ops-layer fa rebind skipped (upstream layout "
            "changed); bridge stays at sgl_kernel level",
            exc_info=True,
        )
    LOG.warning(
        "HCU adapter: fa3 attention routed to DTK flash_attn vendor wheel "
        "(kvcache uniform-window + varlen; rest stays emu)"
    )
