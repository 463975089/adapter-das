# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""SGLang Mamba state-copy launch routing; the upstream wrapper stays intact."""

import hashlib
import importlib
import inspect
import logging

LOG = logging.getLogger("hcu.adapter")
_MODULE = "sglang.kernels.ops.mamba.mamba_state_scatter_triton"
_SYMBOL = "_fused_mamba_state_scatter_with_mask_kernel"
_UPSTREAM_SHA256 = "e4b3b5206d2a28489ce3b422b7f60d0ae811ca2bd603cf97b79f687dfef12795"
_ARGUMENTS = (
    "src_ptr", "dst_ptr", "dst_indices_raw_ptr", "step_indices_raw_ptr",
    "src_indices_raw_ptr", "elem_per_entry", "src_layer_stride",
    "src_req_stride", "src_step_stride", "dst_layer_stride", "dst_req_stride",
    "src_req_size", "src_step_size", "dst_req_size", "BLOCK_SIZE",
)
_WRAPPER_ARGUMENTS = (
    "dst", "src", "dst_indices_raw", "step_indices_raw", "src_indices_raw",
)


class _GridDispatcher:
    """Retain the original JIT fallback and all aliases of its Python caller."""

    def __init__(self, original, replacement, torch_api):
        self.original = original
        self.replacement = replacement
        self.torch = torch_api
        self.device_arches = {}
        self.fast_calls = 0
        self.stock_calls = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _eligible(self, args):
        if len(args) < 6:
            return False
        src, dst = args[:2]
        if not (getattr(src, "is_cuda", False) and getattr(dst, "is_cuda", False)):
            return False
        if src.device != dst.device or src.dtype != self.torch.float32 or dst.dtype != self.torch.float32:
            return False
        # Smaller copies and other storage types retain their original launch.
        if not isinstance(args[5], int) or args[5] < 1024:
            return False
        device = src.device
        if device not in self.device_arches:
            try:
                props = self.torch.cuda.get_device_properties(device)
                arch = props.gcnArchName.split(":")[0]
            except Exception:
                arch = None
            self.device_arches[device] = arch
        return self.device_arches[device] == "gfx936"

    def __getitem__(self, grid):
        stock = self.original[grid]
        if (not isinstance(grid, (tuple, list)) or len(grid) != 3
                or not all(isinstance(x, int) and x > 0 for x in grid)
                or grid[0] * grid[2] > 2147483647):
            return stock

        def launch(*args, **kwargs):
            if not self._eligible(args):
                self.stock_calls += 1
                return stock(*args, **kwargs)
            if self.fast_calls == 0:
                LOG.warning("HCU adapter: Mamba state scatter executing flattened "
                            "grid %s -> %s; source-row mapping retained",
                            tuple(grid), (grid[0] * grid[2], grid[1], 1))
            self.fast_calls += 1
            # Never retry the stock kernel after a launched candidate fails.
            return self.replacement[(grid[0] * grid[2], grid[1], 1)](*args, **kwargs)

        return launch


def _bind(module, replacement, torch_api, register):
    original = getattr(module, _SYMBOL)
    if isinstance(original, _GridDispatcher):
        return original
    wrapper = module.fused_mamba_state_scatter_with_mask
    if tuple(inspect.signature(wrapper).parameters) != _WRAPPER_ARGUMENTS:
        raise RuntimeError("Mamba scatter: upstream wrapper ABI drift")
    if wrapper.__globals__.get(_SYMBOL) is not original:
        raise RuntimeError("Mamba scatter: foreign wrapper namespace")
    if tuple(getattr(original, "arg_names", ())) != _ARGUMENTS:
        raise RuntimeError("Mamba scatter: upstream JIT ABI drift")
    source = getattr(original, "src", "")
    if hashlib.sha256(source.encode()).hexdigest() != _UPSTREAM_SHA256:
        raise RuntimeError("Mamba scatter: unreviewed upstream kernel")
    dispatch = _GridDispatcher(original, replacement, torch_api)
    setattr(module, _SYMBOL, dispatch)
    try:
        register(
            "mamba_state_scatter", replacement,
            target=_MODULE + "." + _SYMBOL, mechanism="rebind",
            detail="gfx936 FP32 entries >=1024; flattened request/block grid; "
                   "upstream wrapper, source-row mapping and cached aliases retained; "
                   "other contracts use original JIT",
        )
    except BaseException:
        setattr(module, _SYMBOL, original)
        raise
    return dispatch


def install():
    from hcu_adapter import kill_switch, registry

    if kill_switch.op_disabled("mamba_state_scatter"):
        LOG.warning("HCU adapter: Mamba state scatter kept upstream "
                    "(HCU_DISABLE_OPS=mamba_state_scatter; startup only)")
        return False
    import torch
    module = importlib.import_module(_MODULE)
    from hcu_adapter.kernels.mamba_state_scatter_triton import (
        _fused_mamba_state_scatter_flat_kernel,
    )

    try:
        dispatch = _bind(module, _fused_mamba_state_scatter_flat_kernel, torch,
                         registry.register)
    except RuntimeError:
        LOG.warning("HCU adapter: Mamba state scatter kept upstream after "
                    "contract preflight refusal", exc_info=True)
        return False
    LOG.warning("HCU adapter: Mamba state scatter flattened-grid route installed; "
                "gfx936 FP32 only; no source writes")
    return dispatch
