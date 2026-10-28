"""Cold-JIT watchdog floor (box16 cold-boot fatality, 2026-10-25).

A cold container compiles the whole JIT surface at first capture/warmup:
8 ranks x sgl-kernel/Triton cold compile outlasts sglang's default
``watchdog_timeout=300`` s -> SIGQUIT kills the tree mid-warmup (box16
first 16-concurrent request died at exactly 300 s wall). Seeded/warm
caches never trip it — only cold boxes need the headroom.

Rule: when the JIT caches look cold (fewer than ~200 files under
TRITON_CACHE_DIR / TORCH_EXTENSIONS_DIR) AND the operator left the
watchdog at its default, raise the floor to 1500 s (proven value on
box16: the cold compile storm fits inside it). An explicit
``--watchdog-timeout`` (any value != the default) always wins.
``HCU_JIT_WATCHDOG=0`` opts out (registry entry; docs/env-vars.md).

Hook points, first that exists wins:
  * 0.5.18-family images — the watchdog factory consumed at scheduler
    init (``scheduler_components/invariant_checker.create_scheduler_
    watchdog``, called with ``watchdog_timeout=get_device().watchdog_
    timeout``);
  * other trees — ``ServerArgs.__init__`` post-hook (field lives on the
    args object).
"""

import itertools
import logging
import os
import pathlib

LOG = logging.getLogger("hcu.adapter")

_DEFAULT_S = 300.0  # sglang's stock watchdog_timeout
_FLOOR_S = 1500.0  # proven headroom for the 8-rank cold compile storm
_COLD_FILES = 200  # a seeded cache has thousands of files; cold ~0


def _count_files(root, cap=_COLD_FILES * 2):
    """Bounded file count under root (rglob is lazy; islice caps work)."""
    try:
        gen = (p for p in pathlib.Path(root).rglob("*") if p.is_file())
        return len(list(itertools.islice(gen, cap)))
    except OSError:
        return 0


def caches_cold():
    """True when every known JIT cache root is thin (or absent)."""
    roots = [
        os.environ.get("TRITON_CACHE_DIR")
        or os.path.expanduser("~/.triton/cache"),
        os.environ.get("TORCH_EXTENSIONS_DIR")
        or os.path.expanduser("~/.cache/torch_extensions"),
    ]
    return max(_count_files(r) for r in roots) < _COLD_FILES


def maybe_bump(value, soft=False):
    """-> (value, bumped). Raise the floor only when the operator left
    the sglang default AND the caches look cold; anything explicit wins."""
    if soft or value is None:
        return value, False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value, False
    if v != _DEFAULT_S:
        return value, False
    if not caches_cold():
        return value, False
    return _FLOOR_S, True


_logged = False


def _log_bump():
    global _logged

    if not _logged:
        _logged = True
        LOG.warning(
            "HCU adapter: cold JIT caches detected -> watchdog floor "
            "raised 300s -> %.0fs (compile-storm headroom; explicit "
            "--watchdog-timeout always wins; HCU_JIT_WATCHDOG=0 opts out)",
            _FLOOR_S,
        )


def install():
    from hcu_adapter import hcu_env

    if not hcu_env.enabled("HCU_JIT_WATCHDOG"):
        return
    hook = _wrap_factory() or _wrap_server_args()
    if hook is None:
        LOG.warning(
            "HCU adapter: cold-JIT watchdog floor not installed "
            "(no hook point on this sglang)"
        )
        return
    try:
        from hcu_adapter import registry

        registry.register(
            "platform_detect",
            "jit-watchdog-floor",
            target=hook,
            mechanism=registry.MECH_REBIND,
            detail=f"cold caches -> watchdog {_DEFAULT_S:.0f}s -> {_FLOOR_S:.0f}s",
        )
    except Exception:  # catalog/governance drift: floor still installed
        pass


def _wrap_factory():
    """0.5.18-family: wrap the factory BEFORE scheduler.py from-imports it
    (sitecustomize runs first, so the scheduler's binding is the wrap)."""
    try:
        from sglang.srt.managers.scheduler_components import (
            invariant_checker as ic,
        )
    except Exception:
        return None
    orig = getattr(ic, "create_scheduler_watchdog", None)
    if orig is None or getattr(orig, "_hcu_wd_wrap", False):
        return None

    def create_watchdog_wrapped(*a, **kw):
        soft = kw.get("soft", a[2] if len(a) > 2 else False)
        if "watchdog_timeout" in kw:
            val, bumped = maybe_bump(kw["watchdog_timeout"], soft)
            kw["watchdog_timeout"] = val
        elif len(a) > 1:
            val, bumped = maybe_bump(a[1], soft)
            a = (a[0], val) + a[2:]
        else:
            bumped = False
        if bumped:
            _log_bump()
        return orig(*a, **kw)

    create_watchdog_wrapped._hcu_wd_wrap = True
    ic.create_scheduler_watchdog = create_watchdog_wrapped
    return (
        "sglang.srt.managers.scheduler_components.invariant_checker"
        ".create_scheduler_watchdog"
    )


def _wrap_server_args():
    """Other trees: post-init bump on the args object that carries the
    field (server_args.ServerArgs)."""
    try:
        from sglang.srt.server_args import ServerArgs
    except Exception:
        return None
    orig_init = ServerArgs.__init__
    if getattr(orig_init, "_hcu_wd_wrap", False):
        return None

    def init_wrapped(self, *a, **kw):
        r = orig_init(self, *a, **kw)
        val, bumped = maybe_bump(getattr(self, "watchdog_timeout", None))
        if bumped:
            self.watchdog_timeout = val
            _log_bump()
        return r

    init_wrapped._hcu_wd_wrap = True
    ServerArgs.__init__ = init_wrapped
    return "sglang.srt.server_args.ServerArgs.__init__"
