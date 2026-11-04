# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from hcu_adapter import hcu_config as config, registry
from hcu_adapter.ops import selector


def load():
    state = config.load()
    options = dict(state.get("hosts", {}).get("deepinfer", {}))
    path = state["path"]
    if "artifacts" in options:
        value = options["artifacts"]
        if not isinstance(value, str) or not value:
            raise ValueError("DeepInfer artifacts must be a nonempty path")
        base = Path(path).resolve().parent if path else Path.cwd()
        options["artifacts"] = str((base / Path(value).expanduser()).resolve())
    op = "deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous"
    if op in state["ops"] and "grouped_gemm_backend" not in options:
        options["grouped_gemm_backend"] = selector.select(op, "triton")
    registry.register("runtime.configuration", "hcu_adapter.bridges._deepinfer.configuration.load",
                      "HCU_CONFIG/config file", "seam")
    return options, {"path": path, "host": "deepinfer"}
