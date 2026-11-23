# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""NCCL_TOPO_FILE card-count and host-PCI mismatch guard (box16 4-proc fatality, 2026-10-25).

Team images preset NCCL_TOPO_FILE to the box's full-8-card topology XML
(validation-host-26 sample: /usr/local/built-in-508-topo-input-tj-default.xml, GPUs
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
  equal count with foreign GPU PCI addresses -> drop the env;
  match    -> keep silently when detected host PCI addresses also match;
  anything unreadable/undetectable -> keep + debug log (conservative).

Debug tier: ``HCU_NCCL_TOPO_GUARD=0`` opts out (registry entry; explicit
env wins over the auto rule, docs/env-vars.md).
"""

import logging
import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET

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


def _topo_busids(topo_path):
    try:
        root = ET.parse(topo_path).getroot()
    except (OSError, ET.ParseError):
        return set()
    return {
        node.attrib["busid"].lower()
        for node in root.iter("pci")
        if node.find("gpu") is not None
        and re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", node.attrib.get("busid", ""))
    }


def _host_gpu_busids(sysfs_root="/sys/class/drm"):
    try:
        return {
            bus
            for node in Path(sysfs_root).glob("renderD*")
            if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", bus := (node / "device").resolve().name.lower())
        }
    except OSError:
        return set()


def _check(topo_path, visible):
    """-> (action, reason): action in {'keep', 'clear'}."""
    topo = _topo_gpus(topo_path)
    if topo <= 0 or visible <= 0:
        return "keep", f"undecidable (topo gpus={topo}, visible={visible})"
    if topo == visible:
        buses = _topo_busids(topo_path)
        host_buses = _host_gpu_busids() if buses else set()
        if host_buses and buses - host_buses:
            return "clear", f"preset topo GPU PCI addresses absent from host: {sorted(buses - host_buses)}"
        return "keep", f"match ({topo} gpus)"
    return (
        "clear",
        f"preset topo describes {topo} gpus but {visible} visible",
    )


def install():
    global _registered

    from hcu_adapter.infra import hcu_env

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
                from hcu_adapter.infra import registry

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
