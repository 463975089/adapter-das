"""Request-boundary state snapshot probe (R-01/R-06 carrier localization).

2026-10-18 forensics: eager prefill non-determinism is CROSS-REQUEST state
leakage (B1==B2 bit-identical, decode replay bit-identical, but Berlin
prefill AFTER Tokyo deviates qv 1.0-3.5). Candidate carriers found by code
audit of /sglang-v0521:
  - DSATokenToKVPool._compress_tail_k/_compress_tail_score: request-scoped
    kpool tail rows (req_pool_size, kpool+extra, 128) per index layer,
    zeroed ONLY at pool construction; no reset anywhere on request
    free/realloc. s1 probe CONFIRMED dirty at every entry after the first
    (tail_k ~6e3 / tail_score ~3.6e3 on req slot 16).
  - MambaPool temporal/conv states: upstream deferred chain is
    alloc(mamba_needs_clear=True) -> prepare_for_extend collects (gated on
    any(holds_mamba), True for fresh reqs) -> ModelRunner
    _maybe_execute_deferred_mamba_cow_and_clear -> clear_slots on the
    forward stream BEFORE model.forward. The s1 probe measured the wrong
    axis (layers instead of slots) — this revision slices [:, idx] and
    additionally witnesses clear_slots calls (indices + pre/post sums) to
    decide whether the chain actually fires in this serving config.

Modes:
- HCU_STATE_SNAPSHOT=1  probe only (JSONL per extend entry + clear witness)
- HCU_STATE_SNAPSHOT=1 HCU_STATE_ZERO=tail|mamba|all  additionally zero the
  chosen carrier rows at req-slot alloc time (register_on_alloc_rows) —
  the flip-rate A/B arm for causality (marathon vs 4/20 baseline).

Path: HCU_STATE_SNAPSHOT_PATH (default /root/state_snap.jsonl).
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import time

import torch

logger = logging.getLogger("hcu.adapter")

_TARGET_MODULE = "sglang.srt.models.glm5_next"
_TARGET_CLASSES = ("Glm5NextModel",)
_POOL_MODULE = "sglang.srt.mem_cache.memory_pool"


def _find_forward_batch(args, kwargs):
    for cand in list(args) + list(kwargs.values()):
        if cand is None or not isinstance(cand, object):
            continue
        if hasattr(cand, "forward_mode") and hasattr(cand, "req_pool_indices"):
            return cand
    return None


def _sum_rows(buf, idx, slot_axis=0):
    """abs-sum of buf rows selected by slot indices along slot_axis."""
    if buf is None or buf.numel() == 0 or idx is None or len(idx) == 0:
        return 0.0
    idx = idx.to(torch.long)
    rows = buf[:, idx] if slot_axis == 1 else buf[idx]
    return float(rows.abs().sum().item())


class _Snap:
    def __init__(self):
        self.path = os.environ.get("HCU_STATE_SNAPSHOT_PATH", "/root/state_snap.jsonl")
        self.zero_mode = os.environ.get("HCU_STATE_ZERO", "")  # tail|mamba|all
        self.topo_done = False
        self.rank = None
        self._zero_hook_done = False

    def _hook_done(self):
        return self._zero_hook_done

    def _rank_ok(self):
        if self.rank is None:
            self.rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
        return self.rank == 0

    def _topo(self, tp, dsa, mp):
        if self.topo_done:
            return
        self.topo_done = True
        info = {
            "t": "topo",
            "tp_type": type(tp).__name__,
            "dsa_type": type(dsa).__name__ if dsa is not None else None,
            "mamba_type": type(mp).__name__ if mp is not None else None,
            "zero_mode": self.zero_mode or None,
        }
        tk = getattr(dsa, "_compress_tail_k", None) if dsa is not None else None
        if tk:
            info["tail_k_shape"] = list(tk[0].shape)
            info["tail_k_layers"] = sum(1 for b in tk if b.numel() > 0)
        mc = getattr(mp, "mamba_cache", None) if mp is not None else None
        if mc is not None:
            info["temporal_shape"] = list(mc.temporal.shape)
            c0 = getattr(mc, "conv", None)
            if c0:
                info["conv_shape"] = list(c0[0].shape)
        with open(self.path, "a") as f:
            f.write(json.dumps(info) + "\n")
        print(f"[HCU state-snap] topology: {info}", flush=True)

    # ---- carrier accessors -------------------------------------------------
    @staticmethod
    def _pools():
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            get_token_to_kv_pool,
        )

        tp = get_token_to_kv_pool()
        dsa = getattr(tp, "full_kv_pool", None)
        mp = getattr(tp, "mamba_pool", None)
        return tp, dsa, mp

    def _mamba_idx(self, req):
        from sglang.srt.model_executor.forward_context import get_req_to_token_pool

        rtp = get_req_to_token_pool()
        if not hasattr(rtp, "get_mamba_indices"):
            return None
        midx = rtp.get_mamba_indices(req)
        tr = getattr(rtp, "translate_mamba_indices", None)
        if tr is not None:
            midx = tr(midx)
        return midx

    def _tail_sums(self, dsa, req):
        tk = getattr(dsa, "_compress_tail_k", None) if dsa is not None else None
        ts = getattr(dsa, "_compress_tail_score", None) if dsa is not None else None
        out = {}
        if tk:
            tot, nz = 0.0, 0
            for buf in tk:
                if buf.numel() == 0:
                    continue
                s = _sum_rows(buf, req, slot_axis=0)
                tot += s
                nz += int(s > 0.0)
            out["tail_k_abs"] = tot
            out["tail_k_nz_layers"] = nz
        if ts:
            tot = 0.0
            for buf in ts:
                if buf.numel() == 0:
                    continue
                tot += _sum_rows(buf, req, slot_axis=0)
            out["tail_score_abs"] = tot
        return out

    def _mamba_sums(self, mp, midx):
        out = {}
        mc = getattr(mp, "mamba_cache", None) if mp is not None else None
        if mc is not None and midx is not None and len(midx):
            # layout: (num_layers, num_slots, ...) -> slot axis 1
            out["mamba_temporal_abs"] = _sum_rows(
                mc.temporal, midx, slot_axis=1
            )
            conv = getattr(mc, "conv", None) or []
            cs = 0.0
            for buf in conv:
                cs += _sum_rows(buf, midx, slot_axis=1)
            out["mamba_conv_abs"] = cs
        return out

    # ---- probe entry --------------------------------------------------------
    def snap(self, fb):
        if not self._rank_ok():
            return
        mode = str(getattr(fb.forward_mode, "forward_mode", fb.forward_mode))
        if "EXTEND" not in mode.upper():
            return
        if torch.cuda.is_current_stream_capturing():
            return
        try:
            tp, dsa, mp = self._pools()
        except Exception:
            logger.warning("hcu state-snap: pool access failed", exc_info=True)
            return
        self._topo(tp, dsa, mp)

        req = fb.req_pool_indices
        if req is None or len(req) == 0:
            return

        rec = {
            "t": "snap",
            "ts": round(time.time(), 3),
            "mode": mode,
            "req": [int(x) for x in req.tolist()],
            "ext": [int(x) for x in fb.extend_seq_lens_cpu]
            if fb.extend_seq_lens_cpu is not None
            else None,
        }
        try:
            rec.update(self._tail_sums(dsa, req))
        except Exception:
            logger.warning("hcu state-snap: tail rows failed", exc_info=True)
        try:
            midx = self._mamba_idx(req)
            rec.update(self._mamba_sums(mp, midx))
        except Exception:
            logger.warning("hcu state-snap: mamba rows failed", exc_info=True)

        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- A/B zero arm -------------------------------------------------------
    def zero_now(self, fb):
        """Zero chosen carrier rows for THIS batch's req slots, at forward
        entry (after the snapshot) — before any layer can read/write them.

        register_on_alloc_rows turned out to never fire on the hybrid pool's
        alloc path (s3tail forensics: hook registered, 0 invocations), so the
        zero point moved here; all TP ranks must zero their local buffers.
        """
        mode = self.zero_mode
        if not mode:
            return
        mstr = str(getattr(fb.forward_mode, "forward_mode", fb.forward_mode))
        if "EXTEND" not in mstr.upper():
            return  # never touch state mid-decode or verify
        ext = fb.extend_seq_lens_cpu
        seq = fb.seq_lens_cpu
        if ext is None or seq is None or len(ext) != len(seq):
            return
        # fresh prefill only: continuing chunks carry a live pending group
        if any(int(s) != int(e) for s, e in zip(seq, ext)):
            return
        want_tail = mode in ("tail", "all")
        want_mamba = mode in ("mamba", "all")
        want_pages = mode in ("pages",)
        zeroed = {}
        try:
            tp, dsa, mp = self._pools()
            req = fb.req_pool_indices
            if req is None or len(req) == 0:
                return
            idx = req.to(torch.long)
            if want_pages:
                loc = getattr(fb, "out_cache_loc", None)
                if loc is not None and len(loc):
                    spp = int(getattr(tp, "slots_per_page", 64) or 64)
                    pages = torch.unique((loc.to(torch.long) // spp))
                    for pool_obj, attr, tag in (
                        (dsa, "k_buffer", "kv_k"),
                        (dsa, "v_buffer", "kv_v"),
                        (
                            getattr(getattr(dsa, "index_key_cache", None), "buffer", None),
                            None,
                            "index",
                        ),
                    ):
                        if pool_obj is None:
                            continue
                        raw = pool_obj if attr is None else getattr(pool_obj, attr, None)
                        if raw is None:
                            continue
                        # k/v buffers are per-layer lists; index buffer may be
                        # a facade tensor -- handle both
                        iter_bufs = raw if isinstance(raw, (list, tuple)) else [raw]
                        for buf in iter_bufs:
                            if (
                                not torch.is_tensor(buf)
                                or buf.numel() == 0
                                or buf.dim() < 2
                            ):
                                continue
                            if buf.shape[0] <= int(pages.max()):
                                continue  # layout mismatch; skip loudly
                            buf[pages] = 0
                            zeroed[tag] = zeroed.get(tag, 0) + 1
            if want_tail and dsa is not None:
                for bufs in (
                    getattr(dsa, "_compress_tail_k", None) or [],
                    getattr(dsa, "_compress_tail_score", None) or [],
                ):
                    for buf in bufs:
                        if buf.numel() == 0:
                            continue
                        buf[idx] = 0
            if want_mamba and mp is not None:
                mc = getattr(mp, "mamba_cache", None)
                if mc is not None:
                    midx = self._mamba_idx(req)
                    if midx is not None and len(midx):
                        mc.temporal[:, midx] = 0
                        for buf in getattr(mc, "conv", None) or []:
                            buf[:, midx] = 0
            if self._rank_ok():
                with open(self.path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "t": "zero",
                                "req": [int(x) for x in idx.tolist()],
                                **{f"zeroed_{k}": v for k, v in zeroed.items()},
                            }
                        )
                        + "\n"
                    )
        except Exception:
            logger.warning("hcu state-snap: zero_now failed", exc_info=True)

    def install_zero_hook(self):  # kept for API compat; no-op registration
        return

    # ---- clear_slots witness --------------------------------------------------
    def install_clear_witness(self):
        from sglang.srt.mem_cache.memory_pool import MambaPool

        orig = MambaPool.clear_slots

        def _witness(self, indices):
            rec = {"t": "clear_witness", "n_idx": int(len(indices))}
            try:
                if _SNAP is not None and _SNAP._rank_ok():
                    mc = getattr(self, "mamba_cache", None)
                    if mc is not None and len(indices):
                        rec["pre_temporal"] = _sum_rows(
                            mc.temporal, indices, slot_axis=1
                        )
                        rec["pre_conv"] = sum(
                            _sum_rows(b, indices, slot_axis=1)
                            for b in (getattr(mc, "conv", None) or [])
                        )
            except Exception:
                pass
            ret = orig(self, indices)
            try:
                if _SNAP is not None and _SNAP._rank_ok():
                    mc = getattr(self, "mamba_cache", None)
                    if mc is not None and len(indices):
                        rec["post_temporal"] = _sum_rows(
                            mc.temporal, indices, slot_axis=1
                        )
                        rec["post_conv"] = sum(
                            _sum_rows(b, indices, slot_axis=1)
                            for b in (getattr(mc, "conv", None) or [])
                        )
            except Exception:
                pass
            with open(_SNAP.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            return ret

        MambaPool.clear_slots = _witness
        print("[HCU state-snap] MambaPool.clear_slots witness installed", flush=True)


_SNAP = None


def install() -> None:
    global _SNAP
    if _SNAP is not None:
        return
    _SNAP = _Snap()

    # clear witness + zero hook land when the pool module imports naturally
    orig_import = importlib.import_module
    _pool_patched = [False]

    def _import(name, *a, **k):
        mod = orig_import(name, *a, **k)
        try:
            if name == _POOL_MODULE and not _pool_patched[0]:
                _pool_patched[0] = True
                _SNAP.install_clear_witness()
                if _SNAP.zero_mode:
                    # pools exist by the time memory_pool is imported? no --
                    # defer hook registration to first forward instead
                    pass
            if name == _TARGET_MODULE and not getattr(
                mod, "_hcu_state_snap_patched", False
            ):
                mod._hcu_state_snap_patched = True
                for cls_name in _TARGET_CLASSES:
                    cls = getattr(mod, cls_name, None)
                    if cls is None:
                        continue
                    orig_fwd = cls.forward

                    def _fwd(self, *args, **kwargs):
                        try:
                            fb = _find_forward_batch(args, kwargs)
                            if fb is not None:
                                _SNAP.snap(fb)
                                if _SNAP.zero_mode:
                                    _SNAP.zero_now(fb)
                        except Exception:
                            logger.warning(
                                "hcu state-snap: snapshot failed", exc_info=True
                            )
                        return orig_fwd(self, *args, **kwargs)

                    cls.forward = _fwd
                    print(
                        f"[HCU state-snap] patched {cls_name}.forward "
                        f"-> {os.path.basename(_SNAP.path)}",
                        flush=True,
                    )
        except Exception:
            logger.warning("hcu state-snap: patch failed", exc_info=True)
        return mod

    importlib.import_module = _import
    print(
        f"[HCU state-snap] installed (path={_SNAP.path}, zero={_SNAP.zero_mode or 'off'})",
        flush=True,
    )
