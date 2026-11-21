# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Startup selection shared by the V4.1 native PCP RoPE providers."""
from functools import lru_cache
import os

from hcu_adapter.infra import hcu_config, hcu_env
from hcu_adapter.infra.kill_switch import op_disabled


MODE_VARIABLE = "HCU_VLLM_PCP_ROPE_IMPL"


@lru_cache(maxsize=8)
def _process_arch(_pid):
    # Policy health checks also run during forward. Probe the device once per
    # process rather than asking the HIP runtime from every attention layer.
    return hcu_env.arch_name()


def _compatible_route():
    from .pcp_fork import selected_mode as pcp_mode

    pcp_enabled = os.environ.get("VLLM_HCU_DSV41_PCP_ATTENTION", "")
    return (pcp_enabled.strip().lower() in {"1", "true", "yes", "on"}
            and pcp_mode() == "native" and _process_arch(os.getpid()) == "gfx938")


def selected_mode(op):
    if op_disabled("vllm_pcp") or op_disabled(op):
        return "stock"
    mode, _ = hcu_env.resolve(MODE_VARIABLE, host="vllm")
    if mode not in ("auto", "hcu", "stock"):
        raise ValueError(f"{MODE_VARIABLE} must be auto, hcu or stock")
    if mode == "auto":
        # Preserve explicit per-operator settings in an existing config file.
        mode = hcu_config.load()["ops"].get(op, "hcu")
        if mode not in ("hcu", "stock"):
            raise ValueError(f"ops.{op} must be stock or hcu")
    return "hcu" if mode == "hcu" and _compatible_route() else "stock"
