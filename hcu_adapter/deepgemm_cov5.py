# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Optional, version-pinned code-object ABI repair for DTK W4A8 HIPC.

The separately built extension must be explicitly present on PYTHONPATH.
Its native INT8/packed-INT4 kernel replaces one low-level call; checkpoint
loading, quantization, scales and every other DeepGEMM operator stay native.
"""

import hashlib
import importlib
from importlib import metadata, util
import inspect
import logging
from pathlib import Path

from . import registry


MODULE = "hcu_deepgemm_w4a8_cov5"
SOURCE_COMMIT = "b061ec70272a001fca622ea814d3ee163e24bb05"
WHEEL_VERSION = "2.1.0+dtk2610.torch2110.2611031138.gb061ec"
WRAPPER_SHA256 = "854157c0752eac436e9cfb31b2896dba7c9d5baecc3bfa8bbe745bc63b5df6e4"
LOG = logging.getLogger("hcu.adapter")


class _Cov5Ops:
    def __init__(self, original, extension):
        self.original = original
        self.extension = extension

    def __getattr__(self, name):
        return getattr(self.original, name)

    def m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
        self, input, weight, output, input_scale, weight_scale, indices, mode
    ):
        if mode != 1002:
            raise NotImplementedError("The COV5 HIPC provider supports mode 1002 only")
        import torch

        arch = torch.cuda.get_device_properties(input.device).gcnArchName.split(":")[0]
        if arch != "gfx936":
            raise NotImplementedError(f"The COV5 HIPC provider requires gfx936, got {arch}")
        return self.extension.run(
            input, weight, output, input_scale, weight_scale, indices, mode
        )


def install():
    """Preserve the public function and cached aliases through private globals."""
    if util.find_spec(MODULE) is None:
        return False
    from .kill_switch import op_disabled

    if op_disabled("w4a8_contiguous_gemm"):
        return False
    if metadata.version("deepgemm") != WHEEL_VERSION:
        raise RuntimeError(f"{MODULE} requires DeepGEMM {WHEEL_VERSION}")
    extension = importlib.import_module(MODULE)
    contract = (
        extension.SOURCE_COMMIT,
        extension.CODE_OBJECT_VERSION,
        tuple(extension.SUPPORTED_MODES),
        extension.ARCHITECTURE,
    )
    if contract != (SOURCE_COMMIT, 5, (1002,), "gfx936"):
        raise RuntimeError(f"Unexpected {MODULE} ABI contract: {contract!r}")
    vendor = importlib.import_module("deepgemm")
    function = vendor.m_grouped_w4a8_gemm_nt_contiguous_hipc
    namespace = function.__globals__
    if isinstance(namespace.get("op"), _Cov5Ops):
        return True
    if (
        list(inspect.signature(function).parameters)
        != ["a", "b", "output", "m_indices", "config"]
        or hashlib.sha256(Path(inspect.getfile(function)).read_bytes()).hexdigest()
        != WRAPPER_SHA256
        or namespace.get("op") is not vendor.op
        or namespace.get("DEFAULT_W4A8_CONTIG_CONFIG") != {"MODE": 1006}
    ):
        raise RuntimeError("DeepGEMM contiguous wrapper changed; COV5 repair refused")
    facade = _Cov5Ops(vendor.op, extension)
    registry.register(
        "w4a8_contiguous_gemm", "deepgemm HIPC1002 COV5",
        "deepgemm.m_grouped_w4a8_gemm_nt_contiguous_hipc",
        registry.MECH_REBIND,
    )
    # Keep the function object, signature, validation and all existing aliases.
    # This module-local default does not mutate the shared vendor config dict.
    namespace["DEFAULT_W4A8_CONTIG_CONFIG"] = {"MODE": 1002}
    namespace["op"] = facade
    LOG.warning("HCU DeepGEMM HIPC1002 COV5: %s", extension.__file__)
    return True
