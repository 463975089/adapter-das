# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Route dynamic per-token FP8 input quantization to LightOp (vLLM 0.33).

Where the cost is
-----------------
`FP8ScaledMMLinearKernel.apply_weights`
(`vllm/model_executor/kernels/linear/scaled_mm/ScaledMMLinearKernel.py:139`)
quantizes the activation through its own `QuantFP8`; on HIP
`QuantFP8.forward_hip` (`input_quant_fp8.py:136`) falls through to
`QuantFP8.forward_cuda` (`:83`), which calls `ops.scaled_fp8_quant`
(`vllm/_custom_ops.py:1836`) -> `torch.ops._C.dynamic_per_token_scaled_fp8_quant`.

Measured on the FP8 baseline trace (dp2 rank0, `20261115_184157_eager`):
442 of the 448 quantization calls go through `QuantFP8` (the other 6 are MoE
`_fp8_quantize` and never reach this module). Per call, the
`_custom_ops.py(1836) scaled_fp8_quant` slice costs 36.40us and the GPU kernel
`dynamic_per_token_scaled_fp8_quant_kernel_strided` 9.58us.

What this bridge does
---------------------
Rebinds `QuantFP8.forward_cuda` on the class itself: `forward_hip` calls it as
`QuantFP8.forward_cuda(self, ...)` (class attribute, not `self.`), so one
rebind covers both the direct calls and the HIP fallback. Only when the call
is exactly the dynamic per-token contract does it call LightOp
`lightop.quant.per_token_quant_fp8`; every other shape -- static scale,
`scale_ub`, per-tensor/per-channel/per-group, token padding, non-contiguous or
non-2D input, non-bf16 activation, LightOp import/execution failure -- runs the
untouched original method.

Numerical difference (measured on several GPUs class, single card, 512x1024 bf16)
----------------------------------------------------------------------------------
* normal rows: scale is bit-identical to upstream; 0.06% of the fp8 values
  differ, at most 1 ulp (max |delta| = one fp8 step, p99 relative error 0).
* all-zero rows: upstream returns the clamp floor `1/(448*512)` as the scale,
  LightOp returns 0. Both quantize to 0 and both dequantize to 0.
* rows whose amax is below `FP8_MAX * scale_floor`: upstream clamps the
  scale to its floor, while LightOp uses the raw `amax / FP8_MAX`. Set
  `HCU_LIGHTOP_FP8_AMAX_DIAG=1` to count those rows; the diagnostic is off by
  default and log-once, because it costs one extra reduction per call.

Accounting
----------
`hit` and per-reason fallback counters live in-process; the first fallback logs
immediately (so a silently misconfigured deployment is visible without waiting
for shutdown) and a final `hit=... fallback=...` line is flushed at exit.

Gate: on by default (`HCU_LIGHTOP_PER_TOKEN_FP8=1`). Set `=0` before startup
for the upstream path. Compilation uses upstream quantization. Eligible graph
capture uses LightOp by default with approximate numerical parity. Set
`HCU_LIGHTOP_PER_TOKEN_FP8_GRAPH=0` to use upstream quantization in capture.
The diagnostic runs only on eligible calls outside capture.
"""

from __future__ import annotations

import atexit
import functools
import importlib
import inspect
import logging
import os
import threading

import torch

LOG = logging.getLogger("hcu.adapter")

_TARGET = "vllm.model_executor.layers.quantization.input_quant_fp8"
_TARGET_SYMBOL = _TARGET + ".QuantFP8.forward_cuda"
_MARKER = "_hcu_lightop_per_token_fp8"
_SIGNATURE = ("self", "x", "scale", "scale_ub", "use_triton")
_AMAX_DIAG_ENV = "HCU_LIGHTOP_FP8_AMAX_DIAG"
_GRAPH_ENV = "HCU_LIGHTOP_PER_TOKEN_FP8_GRAPH"

_LOCK = threading.RLock()
_COUNTS: dict[str, int] = {}
_LOGGED: set[str] = set()
_ATEXIT_ARMED = False
_DIAG = {"calls": 0, "rows": 0, "tiny": 0, "floor": 0.0, "amax_floor": 0.0}


def _bump(reason: str) -> None:
    with _LOCK:
        _COUNTS[reason] = _COUNTS.get(reason, 0) + 1


def _counts_line() -> str:
    with _LOCK:
        items = sorted(_COUNTS.items())
        hit = dict(items).get("hit", 0)
    total = sum(_COUNTS.values())
    parts = " ".join(f"{name}={value}" for name, value in items)
    return f"hit={hit} fallback={total - hit} ({parts or 'no calls'})"


def _log_once(key: str, level: int, message: str, *args, **kwargs) -> None:
    with _LOCK:
        if key in _LOGGED:
            return
        _LOGGED.add(key)
    LOG.log(level, message, *args, **kwargs)


def _lightop_fn():
    """Import the LightOp per-token kernel lazily, once per process."""
    from lightop.quant import per_token_quant_fp8

    if not callable(per_token_quant_fp8):
        raise TypeError("lightop.quant.per_token_quant_fp8 is not callable")
    return per_token_quant_fp8


def _diag_rows(x: torch.Tensor, scale_floor: float, amax_floor: float) -> None:
    """Count rows below the upstream clamp floor (diagnostic gate only)."""
    with torch.no_grad():
        amax = x.detach().abs().amax(dim=-1)
        tiny = int((amax < amax_floor).sum().item())
    with _LOCK:
        _DIAG["calls"] += 1
        _DIAG["rows"] += int(x.shape[0])
        _DIAG["tiny"] += tiny
        _DIAG["floor"] = scale_floor
        _DIAG["amax_floor"] = amax_floor


def _flush() -> None:
    LOG.warning("HCU LightOp per-token FP8 counters: %s", _counts_line())
    with _LOCK:
        calls, rows, tiny, scale_floor, amax_floor = (
            _DIAG["calls"], _DIAG["rows"], _DIAG["tiny"], _DIAG["floor"], _DIAG["amax_floor"],
        )
    if calls:
        share = (100.0 * tiny / rows) if rows else 0.0
        LOG.warning(
            "HCU LightOp per-token FP8 amax diag: calls=%d rows=%d "
            "rows_below_floor=%d (%.4f%%) scale_floor=%.6e amax_floor=%.6e",
            calls, rows, tiny, share, scale_floor, amax_floor,
        )


def _arm_atexit() -> None:
    global _ATEXIT_ARMED
    with _LOCK:
        if _ATEXIT_ARMED:
            return
        _ATEXIT_ARMED = True
    atexit.register(_flush)


def _device_ok(x):
    return x.is_cuda and torch.cuda.current_device() == x.device.index


def _ineligible(instance, x, scale, scale_ub, per_token):
    """Return the fallback reason, or None when the LightOp contract holds."""
    if scale is not None:
        return "static_scale"
    if scale_ub is not None:
        return "scale_ub"
    if getattr(instance, "is_group_quant", True):
        return "group"
    if getattr(instance, "group_shape", None) != per_token:
        return "not_per_token"
    if getattr(instance, "num_token_padding", None) is not None:
        return "token_padding"
    if x.dim() != 2:
        return "not_2d"
    if not x.is_contiguous():
        return "non_contiguous"
    if x.dtype != torch.bfloat16:
        return "dtype"
    return None


def _wrap(module, original):
    """Build the replacement; closes over upstream enum/dtype/floor values."""
    per_token = module.GroupShape.PER_TOKEN
    fp8_dtype = module._FP8_DTYPE
    fp8_max = float(getattr(module, "_FP8_MAX", torch.finfo(fp8_dtype).max))
    scale_floor = float(getattr(module, "_FP8_MIN_SCALING_FACTOR", 1.0 / (fp8_max * 512.0)))
    amax_floor = fp8_max * scale_floor
    diag = os.environ.get(_AMAX_DIAG_ENV, "") not in ("", "0")

    graph_enabled = os.getenv(_GRAPH_ENV, "1") == "1"

    @functools.wraps(original)
    def forward_cuda(self, x, scale=None, scale_ub=None, use_triton=False):
        if torch.compiler.is_compiling():
            return original(self, x, scale, scale_ub, use_triton)
        reason = _ineligible(self, x, scale, scale_ub, per_token)
        capturing = False
        if reason is None:
            if not _device_ok(x):
                reason = "device"
            elif x.numel() == 0:
                reason = "empty"
            else:
                capturing = torch.cuda.is_current_stream_capturing()
                if capturing and not graph_enabled:
                    reason = "capture"
        if reason is not None:
            _bump(reason)
            _log_once(
                "fallback:" + reason, logging.WARNING,
                "HCU LightOp per-token FP8 fallback (%s): %s",
                reason, _counts_line(),
            )
            return original(self, x, scale, scale_ub, use_triton)

        if diag and not capturing:
            _diag_rows(x, scale_floor, amax_floor)
        try:
            quantized, scales = _lightop_fn()(x, dtype=fp8_dtype)
        except Exception:  # noqa: BLE001
            if capturing:
                raise
            _bump("exception")
            _log_once(
                "fallback:exception", logging.WARNING,
                "HCU LightOp per-token FP8 fallback (execution error): %s",
                _counts_line(), exc_info=True,
            )
            return original(self, x, scale, scale_ub, use_triton)

        if capturing:
            _log_once("capture-hit", logging.INFO, "HCU LightOp per-token FP8 captured")
        _bump("hit")
        _log_once(
            "hit", logging.INFO,
            "HCU LightOp per-token FP8 first hit: %s", _counts_line(),
        )
        return quantized, scales.reshape(x.shape[0], 1)

    # Consumer-side fusion may bypass this method only if it preserves this
    # exact quantizer contract, including capture and compilation fallbacks.
    forward_cuda._hcu_token_fp8_backend = "lightop"
    forward_cuda._hcu_token_fp8_graph_enabled = graph_enabled
    return forward_cuda


def install(module=None) -> bool:
    """Rebind `QuantFP8.forward_cuda` when the LightOp contract is available."""
    from hcu_adapter.infra import hcu_env

    if not hcu_env.enabled("HCU_LIGHTOP_PER_TOKEN_FP8"):
        LOG.info("HCU LightOp per-token FP8: skipped (%s=0)",
                 "HCU_LIGHTOP_PER_TOKEN_FP8")
        return False

    if module is None:
        try:
            module = importlib.import_module(_TARGET)
        except Exception as error:  # noqa: BLE001 - optional fast path, never fatal
            LOG.warning("HCU LightOp per-token FP8: skipped (import failed: %s)", error)
            return False

    owner = getattr(module, "QuantFP8", None)
    if not isinstance(owner, type):
        LOG.warning("HCU LightOp per-token FP8: skipped (%s.QuantFP8 missing)",
                    _TARGET)
        return False
    if getattr(owner, _MARKER, False):
        return False
    original = owner.__dict__.get("forward_cuda", getattr(owner, "forward_cuda", None))
    if not callable(original):
        LOG.warning("HCU LightOp per-token FP8: skipped (forward_cuda missing)")
        return False
    try:
        parameters = tuple(inspect.signature(original).parameters)[: len(_SIGNATURE)]
    except (TypeError, ValueError) as error:
        LOG.warning("HCU LightOp per-token FP8: skipped (signature: %s)", error)
        return False
    if parameters != _SIGNATURE:
        LOG.warning("HCU LightOp per-token FP8: skipped (forward_cuda signature %s)",
                    parameters)
        return False
    for attribute in ("GroupShape", "_FP8_DTYPE", "_FP8_MIN_SCALING_FACTOR"):
        if not hasattr(module, attribute):
            LOG.warning("HCU LightOp per-token FP8: skipped (%s missing)", attribute)
            return False
    if not hasattr(module.GroupShape, "PER_TOKEN"):
        LOG.warning("HCU LightOp per-token FP8: skipped (GroupShape.PER_TOKEN missing)")
        return False
    try:
        _lightop_fn()
    except Exception as error:  # noqa: BLE001 - optional dependency
        LOG.warning("HCU LightOp per-token FP8: skipped (lightop unavailable: %s)",
                    error)
        return False

    setattr(owner, "forward_cuda", _wrap(module, original))
    setattr(owner, _MARKER, True)
    _arm_atexit()
    LOG.warning(
        "HCU adapter: LightOp per-token FP8 quantization installed on %s "
        "(dynamic per-token only; every other contract uses upstream)",
        _TARGET_SYMBOL,
    )
    return True
