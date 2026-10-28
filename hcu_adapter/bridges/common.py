"""Cross-framework guards that must run before a host imports GPU stacks."""

from __future__ import annotations

import logging

LOG = logging.getLogger("hcu.adapter")


def install() -> None:
    """Install independent environment, topology, and tvm-ffi guards."""
    try:
        from sgl_kernel._adapters import hcu_env

        hcu_env.log_summary()
    except Exception:
        LOG.warning("HCU adapter: environment summary unavailable", exc_info=True)

    try:
        from sgl_kernel._adapters import nccl_topo_guard

        nccl_topo_guard.install()
    except Exception:
        LOG.warning("HCU adapter: nccl topo guard not installed", exc_info=True)

    try:
        from sgl_kernel._adapters import tvmffi_warmup

        tvmffi_warmup.install()
    except Exception:
        LOG.warning(
            "HCU adapter: tvm-ffi dlpack warmup not installed", exc_info=True
        )
