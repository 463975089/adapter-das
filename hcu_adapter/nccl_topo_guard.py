# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""NCCL_TOPO_FILE card-count mismatch guard (box16 4-proc fatality, 2026-10-25).

Team images preset NCCL_TOPO_FILE to the box's full-8-card topology XML
(nmz26 sample: /usr/local/built-in-508-topo-input-tj-default.xml, GPUs
encoded as ``<gpu dev="N" ...>`` nodes). Running a smaller visible set
(``HIP_VISIBLE_DEVICES=0-3`` for a 4-proc/4-card group) makes NCCL walk
XGMI links toward GPU ids beyond the visible set — fatal at the FIRST
collective::

    Add XGMI error: could not find GPU 5000

The working recipes all ``unset NCCL_TOPO_FILE`` by hand; this guard
makes that automatic. Count ``<gpu`` nodes in the preset XML vs the
torch-visible device count:

  mismatch -> drop the env (NCCL falls back to sysfs detection, correct
              for ANY subset) + one log line naming both counts;
  match    -> keep silently (full-card config is what the file is for);
  anything unreadable/undetectable -> keep + debug log (conservative).

Debug tier: ``HCU_NCCL_TOPO_GUARD=0`` opts out (registry entry; explicit
env wins over the auto rule, docs/env-vars.md).
"""

import logging
import os

LOG = logging.getLogger("hcu.adapter")

_registered = False


def _visible_gpus():
    """torch-visible device count (HIP_VISIBLE_DEVICES already applied)."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _topo_gpus(topo_path):
    """GPU node count in an NCCL topology XML; 0 = unreadable/none found."""
    try:
        with open(topo_path, encoding="utf-8", errors="ignore") as f:
            return f.read().count("<gpu ")
    except OSError:
        return 0


def _check(topo_path, visible):
    """-> (action, reason): action in {'keep', 'clear'}."""
    topo = _topo_gpus(topo_path)
    if topo <= 0 or visible <= 0:
        return "keep", f"undecidable (topo gpus={topo}, visible={visible})"
    if topo == visible:
        return "keep", f"match ({topo} gpus)"
    return (
        "clear",
        f"preset topo describes {topo} gpus but {visible} visible",
    )


def install():
    global _registered

    from hcu_adapter import hcu_env

    if not hcu_env.enabled("HCU_NCCL_TOPO_GUARD"):
        return
    topo = os.environ.get("NCCL_TOPO_FILE", "").strip()
    if not topo:
        return  # nothing preset, nothing to guard
    visible = _visible_gpus()
    action, reason = _check(topo, visible)
    if action == "clear":
        os.environ.pop("NCCL_TOPO_FILE", None)
        LOG.warning(
            "HCU adapter: NCCL_TOPO_FILE cleared (%s -> %s) — NCCL falls "
            "back to sysfs topology; HCU_NCCL_TOPO_GUARD=0 to keep the "
            "preset",
            topo,
            reason,
        )
        if not _registered:
            _registered = True
            try:
                from hcu_adapter import registry

                registry.register(
                    "platform_detect",
                    "nccl-topo-guard",
                    target="env:NCCL_TOPO_FILE",
                    mechanism=registry.MECH_REBIND,
                    detail=f"cleared {topo}: {reason}",
                )
            except Exception:  # catalog/governance drift: guard still acts
                pass
    else:
        LOG.debug("HCU adapter: NCCL_TOPO_FILE kept (%s)", reason)
