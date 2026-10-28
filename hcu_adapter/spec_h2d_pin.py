"""Speculative-decode prepare H2D: pinned ring for the KV-lens copies.

Why: ``eagle_prepare_for_decode`` builds per-request ``cur/nxt_kv_lens``
CPU tensors and ships them with ``.to(device, non_blocking=True)``. For
pageable memory ``non_blocking`` is a no-op in torch -- the copy still
blocks the host, and it is stream-ordered BEHIND the previous forward
(the WAR barrier chain the upstream comment itself warns about), so the
scheduler main thread stalls a full pipeline at that line on every spec
decode step. py-spy on the DSV4F MTP server (nmz16, DTK, 2026-10-21):
12/12 decode-phase samples pinned at eagle_utils.py:1150, scheduler at
~150% CPU, 1.29 s/step at bs1 (vs 0.4 s GPU), step wall growing linearly
with batch. A pinned source buffer turns the copy into a true async DMA
enqueue; an 8-deep ring prevents slot reuse before a lagging DMA reads
it.

The needle patch swaps the two ``.to`` lines for ``_hcu_spec_h2d`` calls
and injects the module-level import above the function. Integer metadata
only (KV lens); no numerics touched. ``HCU_SPEC_H2D_PIN=0`` disables.
"""

from __future__ import annotations

import logging
import pathlib

from . import registry as _reg

logger = logging.getLogger("hcu.adapter")

_PKG = "sglang/srt/speculative/eagle_utils.py"
_DETAIL = (
    "spec-decode prepare_for_decode KV-lens H2D via persistent pinned "
    "ring: torch non_blocking is a no-op for pageable memory, so the "
    "stock .to() blocks the host behind the WAR-chained previous "
    "forward -- a full-pipeline stall per spec decode step (py-spy "
    "12/12, DTK 2026-10-21)."
)

_IMPORT_REPL = (
    "from hcu_adapter.spec_h2d_pin import pinned_h2d "
    "as _hcu_spec_h2d\n\n\n"
    "def eagle_prepare_for_decode(batch: ScheduleBatch):"
)
_IMPORT_NEEDLE = "def eagle_prepare_for_decode(batch: ScheduleBatch):"

_CALL_NEEDLE = (
    "    cur_kv_lens_device = cur_kv_lens_cpu.to("
    "device=batch.device, non_blocking=True)\n"
    "    nxt_kv_lens_device = nxt_kv_lens_cpu.to("
    "device=batch.device, non_blocking=True)\n"
)
_CALL_REPL = (
    "    # HCU spec_h2d_pin: pageable non_blocking H2D is a no-op in torch --\n"
    "    # the copy still blocks the host AND is stream-ordered behind the\n"
    "    # previous forward (WAR barrier chain), stalling a full pipeline per\n"
    "    # spec decode step (py-spy 12/12 at this line, DTK 2026-10-21). A\n"
    "    # pinned ring source makes non_blocking a true async DMA enqueue.\n"
    "    cur_kv_lens_device = _hcu_spec_h2d(cur_kv_lens_cpu, batch.device, \"cur\")\n"
    "    nxt_kv_lens_device = _hcu_spec_h2d(nxt_kv_lens_cpu, batch.device, \"nxt\")\n"
)


def _atomic_write(path: pathlib.Path, text: str) -> None:
    # same compile gate as dsv4_ep_bf16: a syntax-breaking needle takes the
    # whole scheduler import chain down with it.
    compile(text, str(path), "exec")
    tmp = path.with_suffix(path.suffix + ".hcu_tmp")
    tmp.write_text(text)
    tmp.replace(path)


def install() -> None:
    from . import hcu_env as _env

    if not _env.enabled("HCU_SPEC_H2D_PIN"):
        return

    import sglang.kernels.jit as _jit_root

    pkg_root = pathlib.Path(_jit_root.__file__).parents[2]
    p = pkg_root / "srt" / "speculative" / "eagle_utils.py"
    text = p.read_text()

    if "HCU spec_h2d_pin" not in text:
        if _CALL_NEEDLE not in text or text.count(_IMPORT_NEEDLE) != 1:
            raise RuntimeError(
                "HCU adapter: spec_h2d_pin needle drifted in "
                f"{_PKG}; re-anchor before upgrading the sglang tree."
            )
        text = text.replace(_IMPORT_NEEDLE, _IMPORT_REPL, 1)
        text = text.replace(_CALL_NEEDLE, _CALL_REPL, 1)
        _atomic_write(p, text)

    _reg.register(
        "platform_detect",
        "spec-h2d-pinned-ring",
        target=f"text:{_PKG}",
        mechanism=_reg.MECH_TEXT,
        detail=_DETAIL,
    )
    logger.warning(
        "HCU adapter: spec_h2d_pin installed (pinned-ring H2D for spec "
        "prepare_for_decode; HCU_SPEC_H2D_PIN=0 disables)"
    )


# ---------------------------------------------------------------------------
# runtime helper (imported by the patched eagle_utils)
# ---------------------------------------------------------------------------

_SLOTS = 8
_RINGS: dict = {}  # (key, dtype) -> [pinned slot tensors], idx


def pinned_h2d(cpu_t, device, key: str = "d"):
    """CPU 1-D int tensor -> device tensor through a persistent pinned ring.

    The pinned source makes ``non_blocking=True`` a real async DMA enqueue
    (pageable sources silently fall back to a blocking, stream-ordered
    copy). Slot depth 8: a slot is reused only after 8 further steps, by
    which time any in-flight DMA from it has long executed.
    """
    import torch

    flat = cpu_t.reshape(-1)
    rk = (key, flat.dtype)
    slots, idx = _RINGS.get(rk, (None, 0))
    if slots is None or slots[0].numel() < flat.numel():
        # grow once (max_running_requests bounds numel); if pinned alloc is
        # unsupported on this stack, say so once and keep stock behavior
        try:
            slots = [
                torch.empty(
                    max(flat.numel(), 8), dtype=flat.dtype, pin_memory=True
                )
                for _ in range(_SLOTS)
            ]
        except Exception:
            logger.warning(
                "HCU spec_h2d_pin: pin_memory unavailable; falling back to "
                "stock pageable .to() (expect the per-step host stall)."
            )
            _RINGS[rk] = (None, 0)
            return flat.to(device=device, non_blocking=True)
        _RINGS[rk] = (slots, 0)
    slots, idx = _RINGS[rk]
    _RINGS[rk] = (slots, (idx + 1) % _SLOTS)
    slot = slots[idx][: flat.numel()]
    slot.copy_(flat)
    return slot.to(device=device, non_blocking=True)
