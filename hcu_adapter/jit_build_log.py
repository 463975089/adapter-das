"""JIT build progress reporter — one line per real ninja compile.

A cold JIT storm (35 modules, ~5 min each, retry x3 on the broken ones)
redirected into a segment log looks exactly like a wedge from the outside:
no per-module progress reaches the gate console or /tmp/l1_<seg>.log while
pytest holds the segment (2026-10-20 user report: "an hour of compiling,
feels stuck / undying"). This wraps the upstream builder --
``sglang.kernels.jit.utils.compile.ninja.build``, identical signature and
layout in the v0.5.21 and v0.5.22 trees -- and emits one line per attempt:

    HCU JIT: compiling sgl_kernel_jit_metadata pid=123 dir=~/.cache/...
    HCU JIT: built sgl_kernel_jit_metadata in 287s -> .../metadata.so
    HCU JIT: FAILED sgl_kernel_jit_dpsk_v4_topk in 510s (retry may re-enter)

Sinks:

    1. stderr (flushed) -- visible for manual debugging; pytest fd-capture
       swallows it under the gate, which is why 2 exists.
    2. the progress journal ``$HCU_JIT_PROGRESS_LOG`` (default
       ``/tmp/hcu_jit_progress.log``; set empty ``=""`` to disable), written
       with O_APPEND + os.write => immune to pytest capture and block
       buffering. The gate heartbeat (ci/l1_gpu_gate.sh run_seg) tails this
       file every 60s, so "compiling vs wedged" is tellable live: compiling
       keeps appending lines, a wedge leaves the last line frozen.

Cache hits never enter build(), so START lines mean real compiles only.
"""

import functools
import logging
import os
import pathlib
import sys
import time

LOG = logging.getLogger("hcu.adapter")

_MARK = "_hcu_build_reported"


def journal_path():
    return os.environ.get("HCU_JIT_PROGRESS_LOG", "/tmp/hcu_jit_progress.log")


def emit(line):
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:  # stderr closed under some harnesses; never fatal
        pass
    path = journal_path()
    if not path:
        return
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (line + "\n").encode("utf-8", "replace"))
        finally:
            os.close(fd)
    except OSError:
        pass  # journal is best-effort observability, never a boot blocker


def wrap(mod):
    """Monkeypatch ``mod.build`` with the reporting wrapper (idempotent)."""
    real = getattr(mod, "build", None)
    if real is None or getattr(real, _MARK, False):
        return mod

    @functools.wraps(real)
    def build(*, spec, build_dir, build_file):
        name = getattr(spec, "module_name", "<unknown>")
        emit(f"HCU JIT: compiling {name} pid={os.getpid()} dir={build_dir}")
        t0 = time.monotonic()
        try:
            so = real(spec=spec, build_dir=build_dir, build_file=build_file)
        except BaseException:
            dt = time.monotonic() - t0
            emit(
                f"HCU JIT: FAILED {name} in {dt:.0f}s "
                f"(ninja detail in RuntimeError above; build_dir kept: {build_dir})"
            )
            raise
        dt = time.monotonic() - t0
        emit(f"HCU JIT: built {name} in {dt:.0f}s -> {so}")
        return so

    setattr(build, _MARK, True)
    mod.build = build
    LOG.info(
        "HCU adapter: JIT build progress reporter installed on %s",
        getattr(mod, "__name__", "<ninja>"),
    )
    return mod


def install():
    try:
        from sglang.kernels.jit.utils.compile import ninja as _ninja
    except ImportError:
        LOG.debug(
            "HCU adapter: JIT build reporter not installed "
            "(sglang.kernels.jit not importable yet)"
        )
        return
    wrap(_ninja)
