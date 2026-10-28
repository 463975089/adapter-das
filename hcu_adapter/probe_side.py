"""In-graph side-buffer probe for the fa3 emu boundary in serving CUDA graphs.

Motivation (2026-10-18, GLM-5.3-Flash TP8+MTP): prefill-verify spec mode
with cuda graphs emits garbage, while the same config without graphs is
green and the emu alone replays bit-exact at verify shapes. Forward hooks
only fire at CAPTURE time, so the only way to observe what a replay
actually feeds the emu is to make the observation part of the graph: a
``copy_`` into a side buffer pre-allocated OUTSIDE the capture region is
recorded like any other op and re-executes on every replay, so reading the
side buffer after ``replay()`` yields the replay-time values.

Modes (HCU_SIDE_PROBE):
  replay -- graph arm: each capture session (a ``torch.cuda.graph`` ctx)
            whose first emu call's batch matches HCU_SIDE_PROBE_BS gets its
            emu inputs/outputs copied into the side slab; after every
            CUDAGraph.replay() the session is dumped to disk. Eager calls
            once serving has started (after the first capture ended) are
            buffered and dumped as well (the prefill).
  eager  -- no-graph arm: every emu call is buffered and dumped in chunks;
            these are the reference values the graph arm must reproduce at
            step 1 before any corruption can compound.

Snapshotted per call: q, qv, cu_seqlens_q, cache_seqlens, page_table,
k/v descale, output o (+ lse when returned), plus python-level flags
(causal, window, scale, max_seqlen_q, page geometry). Small tensors only
(HCU_SIDE_PROBE_MAX_TENSOR_MB) — the KV pool itself is never copied.

Dumps land in HCU_SIDE_PROBE_OUT (default /root/probe_out):
  g<sid>_r<n>_p<pid>.pt   one graph session per replay (replay mode)
  eagerchunk_<k>_p<pid>.pt  32 eager calls per file
  status_p<pid>.log       session arm/capture breadcrumbs

Everything here is debug tooling behind an env gate; sitecustomize imports
this module only when HCU_SIDE_PROBE is set.
"""

import os

import torch

_MODE = os.environ.get("HCU_SIDE_PROBE", "")
_OUT = os.environ.get("HCU_SIDE_PROBE_OUT", "/root/probe_out")
# gate in EMU terms: cache_seqlens rows == total query tokens in the batch
# (e.g. verify graph for a bs=1 request shows 6 rows with draft tokens=6;
# draft decode shows bs rows). Comma-separated list allowed, 0/empty = all.
_PROBE_BS = {
    int(x)
    for x in os.environ.get("HCU_SIDE_PROBE_BS", "0").replace(",", " ").split()
    if x.strip() and x != "0"
}
_SLAB_BYTES = int(os.environ.get("HCU_SIDE_PROBE_SLAB_MB", "256")) << 20
_MAX_TENSOR_BYTES = int(
    float(os.environ.get("HCU_SIDE_PROBE_MAX_TENSOR_MB", "4")) * (1 << 20)
)
_MAX_REPLAYS = int(os.environ.get("HCU_SIDE_PROBE_MAX_REPLAYS", "40"))
_MAX_EAGER = int(os.environ.get("HCU_SIDE_PROBE_MAX_EAGER", "1500"))
_MAX_CALLS_PER_SESSION = int(
    os.environ.get("HCU_SIDE_PROBE_MAX_CALLS_PER_SESSION", "400")
)
_MAX_SESSIONS = int(os.environ.get("HCU_SIDE_PROBE_MAX_SESSIONS", "64"))
_EAGER_CHUNK = 32
# dump window: when set, replay dumps and eager buffering happen only while
# this file exists (cheap stat per call) — avoids paying the per-replay
# torch.save sync on warmup traffic and pins the dumps to the request under
# test. Empty/unset = always on (tests, eager smoke).
_FLAG = os.environ.get("HCU_SIDE_PROBE_FLAG", "")


def _flag_on():
    if not _FLAG:
        return True
    try:
        return os.path.exists(_FLAG)
    except Exception:
        return True

_SLAB = None
_SLICE_OFF = 0  # next free byte in _SLAB
_SESSIONS = 0
_CUR = None  # capture session active on this thread
_SERVE = False  # True once any capture ended -> eager calls are real serving
_EAGER_BUF = []
_EAGER_DUMPS = 0
_LOGGED = set()

_DTYPES = {
    "uint8": torch.uint8, "int8": torch.int8,
    "int32": torch.int32, "int64": torch.int64,
    "float32": torch.float32, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn,
}


def _log_once(tag, exc):
    if tag in _LOGGED:
        return
    _LOGGED.add(tag)
    try:
        import traceback

        with open(os.path.join(_OUT, f"status_p{os.getpid()}.log"), "a") as f:
            f.write(f"[probe_side] {tag}: {exc!r}\n{traceback.format_exc()}\n")
    except Exception:
        pass


def _rank():
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return -1


def _ensure_slab(device):
    global _SLAB
    if _SLAB is None:
        _SLAB = torch.empty(_SLAB_BYTES, dtype=torch.uint8, device=device)
    return _SLAB


def _snap_fields(fn_name, args, kwargs, result):
    """Pick the diagnostic tensors + python flags out of one emu call."""

    def pick(idx, name):
        if name in kwargs:
            return kwargs[name]
        return args[idx] if len(args) > idx else None

    if fn_name == "kvcache":
        # flash_attn_with_kvcache positional order (see sgl_kernel/flash_attn)
        named = [
            ("q", pick(0, "q")),
            ("qv", pick(5, "qv")),
            ("cu_q", pick(12, "cu_seqlens_q")),
            ("cache_seqlens", pick(8, "cache_seqlens")),
            ("page_table", pick(11, "page_table")),
            ("k_descale", pick(17, "k_descale")),
            ("v_descale", pick(18, "v_descale")),
        ]
        k_cache, v_cache = pick(1, "k_cache"), pick(2, "v_cache")
        meta = {
            "fn": "kvcache",
            "causal": bool(kwargs.get("causal", False)),
            "window": tuple(kwargs.get("window_size", (-1, -1))),
            "scale": kwargs.get("softmax_scale"),
            "max_seqlen_q": pick(14, "max_seqlen_q"),
            "only_qv": bool(kwargs.get("only_qv", False)),
            "k_cache_shape": tuple(k_cache.shape) if torch.is_tensor(k_cache) else None,
            "v_cache_shape": tuple(v_cache.shape) if torch.is_tensor(v_cache) else None,
        }
    else:
        # flash_attn_varlen_func
        named = [
            ("q", pick(0, "q")),
            ("cu_q", pick(3, "cu_seqlens_q")),
            ("cu_k", pick(4, "cu_seqlens_k")),
        ]
        meta = {
            "fn": "varlen",
            "causal": bool(kwargs.get("causal", False)),
            "scale": kwargs.get("softmax_scale"),
        }
    if isinstance(result, tuple):
        o, lse = result[0], result[1] if len(result) > 1 else None
        if torch.is_tensor(lse):
            named.append(("lse", lse))
    else:
        o = result
    if torch.is_tensor(o):
        named.append(("o", o))
    fields = [
        (name, t) for name, t in named
        if torch.is_tensor(t)
        and t.device.type == "cuda"
        and t.numel() > 0
        and t.numel() * t.element_size() <= _MAX_TENSOR_BYTES
    ]
    for name, t in fields:
        meta[f"{name}_shape"] = tuple(t.shape)
        meta[f"{name}_dtype"] = str(t.dtype).replace("torch.", "")
    bs = 0
    for name, t in fields:
        if name == "cache_seqlens":
            bs = t.shape[0]
    if not bs:
        for name, t in fields:
            if name == "cu_q":
                bs = t.shape[0] - 1
    meta["bs"] = bs
    return meta, fields


def _record_capture(meta, fields):
    """Register side copies into the slab; the copy_ kernels recorded here
    re-execute on every replay of the capturing graph."""
    global _SESSIONS, _SLICE_OFF
    sess = _CUR
    if sess is None or sess.get("dead"):
        return
    if sess["armed"] is None:
        sess["armed"] = meta["bs"] in _PROBE_BS if _PROBE_BS else True
        if not sess["armed"]:
            sess["dead"] = True
            _status(f"session {sess['id']}: DISARMED (bs={meta['bs']})")
            return
    if sess["idx"] >= _MAX_CALLS_PER_SESSION:
        sess["dead"] = True
        return
    if not fields:
        return
    if _SLAB is None:
        # the slab MUST pre-exist capture: allocating it now would draw
        # from the graph's private pool and later captures would clobber
        # it. Warmup always precedes capture (runner design), so this only
        # fires if that assumption breaks — record nothing, say so loudly.
        _log_once("slab-late", RuntimeError("first call was already capturing"))
        sess["dead"] = True
        return
    entry = dict(meta)
    entry["idx"] = sess["idx"]
    entry["fields"] = {}
    for name, t in fields:
        nbytes = t.numel() * t.element_size()
        off = (sess["off"] + 15) & ~15
        if off + nbytes > _SLAB_BYTES:
            sess["dead"] = True
            _status(f"session {sess['id']}: slab exhausted at call {sess['idx']}")
            break
        dst = _SLAB[off : off + nbytes].view(
            _DTYPES.get(str(t.dtype).replace("torch.", ""), t.dtype)
        )
        dst = dst.view(t.shape)
        dst.copy_(t)  # recorded into the graph
        entry["fields"][name] = {
            "off": off, "nbytes": nbytes,
            "shape": tuple(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
        }
        sess["off"] = off + nbytes
    sess["calls"].append(entry)
    sess["idx"] += 1


def _buffer_eager(meta, fields):
    global _EAGER_DUMPS
    if _EAGER_DUMPS >= _MAX_EAGER:
        return
    entry = dict(meta)
    for name, t in fields:
        entry[name] = t.detach().clone()
    _EAGER_BUF.append(entry)
    if len(_EAGER_BUF) >= _EAGER_CHUNK:
        _flush_eager()


def _flush_eager():
    global _EAGER_DUMPS
    if not _EAGER_BUF:
        return
    k = _EAGER_DUMPS // _EAGER_CHUNK
    payload = {
        "rank": _rank(),
        "base_idx": _EAGER_DUMPS,
        "calls": _EAGER_BUF,
    }
    torch.save(payload, os.path.join(_OUT, f"eagerchunk_{k:04d}_p{os.getpid()}.pt"))
    _EAGER_DUMPS += len(_EAGER_BUF)
    _EAGER_BUF.clear()


def _status(line):
    try:
        with open(os.path.join(_OUT, f"status_p{os.getpid()}.log"), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _dump_session(sess, tag):
    if sess["armed"] is not True or not sess["calls"]:
        return
    calls = []
    for entry in sess["calls"]:
        e = dict(entry)
        fdesc = e.pop("fields")
        for name, fd in fdesc.items():
            t = _SLAB[fd["off"] : fd["off"] + fd["nbytes"]]
            t = t.view(_DTYPES.get(fd["dtype"], torch.uint8)).view(fd["shape"])
            e[name] = t.clone()
        calls.append(e)
    import time as _time

    payload = {
        "rank": _rank(),
        "session": sess["id"],
        "tag": tag,
        "t": _time.time(),
        "calls": calls,
    }
    torch.save(payload, os.path.join(_OUT, f"g{sess['id']}_{tag}_p{os.getpid()}.pt"))


def _wrap(fn_name, fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        try:
            capturing = (
                torch.cuda.is_available()
                and torch.cuda.is_current_stream_capturing()
            )
            meta, fields = _snap_fields(fn_name, args, kwargs, result)
            if capturing:
                _record_capture(meta, fields)
            else:
                if _MODE == "replay" and fields and _SLAB is None:
                    # pre-capture warmup: the one safe moment to draw the
                    # slab from the regular allocator on the right device
                    _ensure_slab(fields[0][1].device)
                if _MODE == "eager" or _SERVE:
                    if _flag_on():
                        _buffer_eager(meta, fields)
                    elif _EAGER_BUF:
                        _flush_eager()
        except Exception as exc:  # never break serving/capture
            _log_once(f"record:{fn_name}", exc)
        return result

    return wrapper


def install():
    if _MODE not in ("replay", "eager"):
        return
    os.makedirs(_OUT, exist_ok=True)
    _status(
        f"probe_side install mode={_MODE} bs={sorted(_PROBE_BS) or 'ALL'} "
        f"slab_mb={_SLAB_BYTES >> 20}"
    )

    import atexit

    atexit.register(_flush_eager)

    import sgl_kernel.flash_attn as _fa

    for attr, fname in (
        ("flash_attn_with_kvcache", "kvcache"),
        ("flash_attn_varlen_func", "varlen"),
    ):
        orig = getattr(_fa, attr)
        if getattr(orig, "_hcu_side_probe", False):
            continue
        wrapped = _wrap(fname, orig)
        wrapped._hcu_side_probe = True
        setattr(_fa, attr, wrapped)

    if _MODE != "replay":
        return

    # capture-session tracking: torch.cuda.graph(g) is a context manager
    import torch.cuda.graphs as _graphs

    _orig_enter = _graphs.graph.__enter__
    _orig_exit = _graphs.graph.__exit__

    def _enter(self):
        global _SESSIONS, _CUR
        r = _orig_enter(self)
        sess = None
        if _SESSIONS < _MAX_SESSIONS:
            sess = {
                "id": _SESSIONS, "armed": None, "dead": False,
                "idx": 0, "off": _SLICE_OFF, "calls": [], "dumps": 0,
            }
            _SESSIONS += 1
        # tag the CUDAGraph (self.cuda_graph) — THAT is the object whose
        # .replay() the runner calls; `self` here is the context manager.
        cg = getattr(self, "cuda_graph", None)
        if cg is not None:
            cg._hcu_probe_sess = sess
        _CUR = sess
        return r

    def _exit(self, *exc):
        global _CUR, _SLICE_OFF
        sess = getattr(getattr(self, "cuda_graph", None), "_hcu_probe_sess", None)
        if sess is not None:
            _SLICE_OFF = max(_SLICE_OFF, sess["off"])
            _status(
                f"session {sess['id']} captured: armed={sess['armed']} "
                f"bs-gate={_PROBE_BS} calls={sess['idx']}"
            )
        _CUR = None
        return _orig_exit(self, *exc)

    _graphs.graph.__enter__ = _enter
    _graphs.graph.__exit__ = _exit

    _orig_replay = torch.cuda.CUDAGraph.replay

    def _replay(self):
        global _SERVE
        _orig_replay(self)
        _SERVE = True  # first replay == real serving; eager calls from here
        sess = getattr(self, "_hcu_probe_sess", None)
        if sess is None:
            return
        try:
            if (
                sess["armed"] is True
                and sess["dumps"] < _MAX_REPLAYS
                and _flag_on()
            ):
                sess["dumps"] += 1
                _dump_session(sess, f"r{sess['dumps']:03d}")
        except Exception as exc:
            _log_once("dump", exc)

    torch.cuda.CUDAGraph.replay = _replay
