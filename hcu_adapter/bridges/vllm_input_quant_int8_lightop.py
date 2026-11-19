# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Route dynamic per-token INT8 input quantization to LightOp (vLLM 0.33).

Where the cost is
-----------------
`TritonInt8ScaledMMLinearKernel.apply_weights`
(`vllm/model_executor/kernels/linear/scaled_mm/triton.py:132`) quantizes the
activation through `ops.scaled_int8_quant(x, i_s, i_zp, symmetric=symmetric)`
(`vllm/_custom_ops.py:1908`), which for the dynamic case allocates the output
and the per-token scales and calls `torch.ops._C.dynamic_scaled_int8_quant`
(`:1959`) -> kernel `vllm::dynamic_scaled_int8_quant_kernel<c10::BFloat16,
float>`.

On the w4a8 trace (`20261117_162243`, dp0 rank0, 7.96 s window) this path runs
2280 times and is the densest quantization in the model: the dense GEMM itself
already goes through LightOp (`hcu_adapter/kernels/scaled_mm_lightop.py` ->
`lightop hipblaslt_w8a8_channelwise_gemm`), and the MoE activation quant is
already `at::native::swiglu_clamp_quant_kernel` from `lightop/op*.so`. Only the
dense activation quant was still upstream.

What this bridge does
---------------------
Rebinds `vllm._custom_ops.scaled_int8_quant`. Every in-tree caller reaches it
as a module attribute (`from vllm import _custom_ops as ops` then
`ops.scaled_int8_quant(...)` -- `scaled_mm/{triton,cutlass,aiter}.py`,
`fused_moe/utils.py`, `quark/quark_moe.py`; there is no `from
vllm._custom_ops import scaled_int8_quant` anywhere), so one rebind covers all
of them.

Only the exact dynamic per-token symmetric INT8 contract is routed to LightOp
`lightop.op.per_token_dynamic_quant_int8` (the native op, not the python
wrapper, so scale layout stays `[M, 1]` like upstream and no extra `[M]`
tensor is allocated). Everything else -- static scale, asymmetric/azp, non-2D,
non-contiguous, non-bf16, hidden size outside the native kernel window,
misaligned pointer, empty input, LightOp import or execution failure -- runs
the untouched original function.

Why this entry point
--------------------
`lightop.quant.per_token_quant_int8` is **not** usable here: its native fast
path is hard-wired to `K == 128` (`csrc/per_token_quant_int8.cu:43`,
`per_token_quant_int8_bf16_k128_kernel`, `TORCH_CHECK(input.size(-1) == 128)`),
so every MLA width falls to its Triton kernel and is 2.7-2.95x slower
(36.4-37.4 us vs 12.4-13.7 us, see `workspace/mla-step3-int8-microbench.md`).
`per_token_dynamic_quant_int8` is generic over `hidden_size % 16 == 0 and
hidden_size <= 8192`, which covers all MLA widths, and with
`smooth_scale == 1` / `round_after_smooth == False` its arithmetic is
`scale = row_max / 127`, `inv = (row_max == 0) ? 0 : 127 / row_max`,
`float_to_int8_rn` -- the same contract as upstream.

Measured (single card, `workspace/probes/int8_quant_ab_v2.py`)
-------------------------------------------------------------
* whole call including python dispatch and allocation: upstream 12.2-13.4 us
  vs LightOp 10.7-11.5 us, ratio **0.861** (0.84-0.91), i.e. ~1.8 us/call.
  Both paths launch exactly one kernel per call.
* numerics are **bit-identical**: scales `torch.equal`, quantized values 0
  differences, including the all-zero row (both scale 0.0) and a tiny-amax row
  (both 3.043e-10). No floor is applied unless `round_after_smooth=True`.
* the gain is in the host wrapper (preallocated direct calls are on par:
  5.8-7.2 us vs 4.6-7.5 us; at hidden 8192 the LightOp kernel is even a bit
  slower: 6.26 us vs 5.43 us). Do not report it as a GPU-side win.

Accounting
----------
Each worker keeps lock-free `hit` and per-reason fallback counters. The first
hit logs at INFO, and the first fallback for each reason logs at WARNING.
The `atexit` hook reports the totals on normal shutdown. Correlate hits with
the LightOp kernel in the trace. Upstream calls can still appear for graph
captures and inputs outside the LightOp contract.

Known limits (validate in the service, do not treat as proven)
--------------------------------------------------------------
* The `ones(hidden)` smooth factor is cached by device, stream and hidden
  width. Compilation and default graph capture use the native op. With
  HCU_LIGHTOP_PER_TOKEN_INT8_GRAPH=1, capture can use a warmed stream factor
  or allocate a graph-local factor without publishing it into the eager cache.
* Accelerator placement is an explicit gate (`_device_ok`): a CPU tensor falls
  back with reason `device` instead of reaching the HCU op and logging an
  exception fallback.

Gate: opt-in, off by default (`HCU_LIGHTOP_PER_TOKEN_INT8=0`); set `=1` to
enable. Numerics do not change, but this is a vendor kernel swap on the
hottest quantization in the model, so it stays behind a switch until the
service A/B (kernel name switch + counters + HumanEval) has been run.
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

_TARGET = "vllm._custom_ops"
_TARGET_SYMBOL = _TARGET + ".scaled_int8_quant"
_ENV = "HCU_LIGHTOP_PER_TOKEN_INT8"
_GRAPH_ENV = "HCU_LIGHTOP_PER_TOKEN_INT8_GRAPH"
_MARKER = "_hcu_lightop_per_token_int8"
_SIGNATURE = ("input", "scale", "azp", "symmetric")

# `per_token_dynamic_quant_int8` dispatches to its generic native kernel only
# for hidden_size % 16 == 0 && 0 < hidden_size <= 8192 and 16-byte aligned
# pointers (csrc/per_token_dynamic_quant_int8.cu:427).
_VEC = 16
_MAX_HIDDEN = 8192

_LOCK = threading.RLock()
_COUNTS: dict[str, int] = {}
_LOGGED: set[str] = set()
_ATEXIT_ARMED = False
_SMOOTH: dict[tuple, torch.Tensor] = {}
_OP = None
_BINDING = None


def _bump(reason: str) -> int:
    """Count one call. Lock-free on purpose: it is on the hot path.

    Each vLLM worker runs the model forward on one thread, so the read-modify
    write below is not contended there; if it ever were, a lost increment only
    perturbs a diagnostic count, never the quantization result.
    """
    count = _COUNTS.get(reason, 0) + 1
    _COUNTS[reason] = count
    return count


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


def _flush() -> None:
    LOG.warning("HCU LightOp per-token INT8 counters: %s", _counts_line())


def _arm_atexit() -> None:
    global _ATEXIT_ARMED
    with _LOCK:
        if _ATEXIT_ARMED:
            return
        _ATEXIT_ARMED = True
    atexit.register(_flush)


def _lightop_op():
    """Return the LightOp native per-token INT8 quant op (cached after first use).

    The import is resolved once and kept in `_OP`: a module-level `import` on
    the hot path costs ~1us per call, which is the same order as the whole
    saving this bridge is after.
    """
    global _OP
    op = _OP
    if op is not None:
        return op
    from lightop import op as lightop_op

    op = getattr(lightop_op, "per_token_dynamic_quant_int8", None)
    if not callable(op):
        raise TypeError(
            "lightop.op.per_token_dynamic_quant_int8 is not callable"
        )
    _OP = op
    return op


def _device_ok(x) -> bool:
    """True when the activation lives on an accelerator (HIP reports as cuda)."""
    return bool(getattr(x, "is_cuda", False))


def _execution_fallback(x):
    if torch.cuda.current_device() != x.device.index:
        return "device"
    return None


def _current_stream_id(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def _smooth_scale(hidden: int, device: torch.device) -> torch.Tensor:
    """Keep each smooth factor on its creating stream outside graph capture."""
    stream = _current_stream_id(device)
    key = (device.index, stream, hidden)
    cached = _SMOOTH.get(key)
    if cached is not None:
        return cached
    cached = torch.ones(hidden, device=device, dtype=torch.float32)
    with _LOCK:
        _SMOOTH.setdefault(key, cached)
        cached = _SMOOTH[key]
    _log_once(
        f"smooth:{device},{stream},{hidden}", logging.INFO,
        "HCU LightOp per-token INT8 smooth cache created: hidden=%d "
        "device=%s stream=%d", hidden, device, stream,
    )
    return cached


def _ineligible(x, scale, azp, symmetric):
    """Return the fallback reason, or None when the LightOp contract holds."""
    if scale is not None:
        return "static_scale"
    if azp is not None:
        return "azp"
    if not symmetric:
        return "asymmetric"
    if not isinstance(x, torch.Tensor):
        return "not_tensor"
    if x.dim() != 2:
        return "not_2d"
    if not x.is_contiguous():
        return "non_contiguous"
    if x.dtype != torch.bfloat16:
        return "dtype"
    hidden = int(x.shape[-1])
    if hidden <= 0 or hidden % _VEC != 0 or hidden > _MAX_HIDDEN:
        return "hidden"
    if int(x.numel() // hidden) == 0:
        return "empty"
    if x.data_ptr() % _VEC != 0:
        return "misaligned"
    if not _device_ok(x):
        return "device"
    return None


def _wrap(original):
    """Build the replacement around the captured upstream function."""

    graph_enabled = os.getenv(_GRAPH_ENV) == "1"

    @functools.wraps(original)
    def scaled_int8_quant(input, scale=None, azp=None, symmetric=True):
        if torch.compiler.is_compiling():
            return original(input, scale, azp, symmetric)
        reason = _ineligible(input, scale, azp, symmetric)
        if reason is None:
            reason = _execution_fallback(input)
        capturing = (reason is None and input.is_cuda
                     and torch.cuda.is_current_stream_capturing())
        if capturing and not graph_enabled:
            reason = "capture"
        if reason is not None:
            _bump(reason)
            _log_once(
                "fallback:" + reason, logging.WARNING,
                "HCU LightOp per-token INT8 fallback (%s): %s",
                reason, _counts_line(),
            )
            return original(input, scale, azp, symmetric)

        # Inlined `_quantize`: one less python frame on the hot path (the frame
        # count also biases `with_stack` profiles) and no tuple round trip.
        # Upstream contract is (quantized, scales[M, 1], azp); allocate both so
        # the op writes straight into the layout the GEMM expects.
        try:
            hidden = int(input.shape[-1])
            tokens = int(input.numel() // hidden)
            if capturing:
                stream = _current_stream_id(input.device)
                smooth = _SMOOTH.get((input.device.index, stream, hidden))
                if smooth is None:
                    smooth = torch.ones(hidden, device=input.device, dtype=torch.float32)
            else:
                smooth = _smooth_scale(hidden, input.device)
            quantized = torch.empty_like(input, dtype=torch.int8)
            scales = torch.empty((tokens, 1), device=input.device,
                                 dtype=torch.float32)
            _lightop_op()(quantized, input, scales, smooth, False, -1.0)
        except Exception:  # noqa: BLE001
            if capturing:
                raise
            _bump("exception")
            _log_once(
                "fallback:exception", logging.WARNING,
                "HCU LightOp per-token INT8 fallback (execution error): %s",
                _counts_line(), exc_info=True,
            )
            return original(input, scale, azp, symmetric)

        if capturing:
            _log_once("capture-hit", logging.INFO, "HCU LightOp per-token INT8 captured")
        if _bump("hit") == 1:
            LOG.info("HCU LightOp per-token INT8 first hit: %s", _counts_line())
        return quantized, scales, None

    return scaled_int8_quant


def native_quant_binding(module):
    """Return the original helper only while this bridge still owns the binding."""
    current = module.scaled_int8_quant
    binding = _BINDING
    if binding is not None and binding[0] is module and binding[1] is current:
        return binding[2]
    return current


def install(module=None) -> bool:
    """Rebind `vllm._custom_ops.scaled_int8_quant` when the contract holds."""
    from hcu_adapter.infra import hcu_env

    global _BINDING

    if not hcu_env.enabled(_ENV):
        LOG.info("HCU LightOp per-token INT8: skipped (%s=0)", _ENV)
        return False

    if module is None:
        try:
            module = importlib.import_module(_TARGET)
        except Exception as error:  # noqa: BLE001 - optional fast path, never fatal
            LOG.warning("HCU LightOp per-token INT8: skipped (import failed: %s)",
                        error)
            return False

    if getattr(module, _MARKER, False):
        return False
    original = getattr(module, "scaled_int8_quant", None)
    if not callable(original):
        LOG.warning("HCU LightOp per-token INT8: skipped (scaled_int8_quant missing)")
        return False
    try:
        parameters = tuple(inspect.signature(original).parameters)[: len(_SIGNATURE)]
    except (TypeError, ValueError) as error:
        LOG.warning("HCU LightOp per-token INT8: skipped (signature: %s)", error)
        return False
    if parameters != _SIGNATURE:
        LOG.warning("HCU LightOp per-token INT8: skipped (scaled_int8_quant "
                    "signature %s)", parameters)
        return False
    try:
        _lightop_op()
    except Exception as error:  # noqa: BLE001 - optional dependency
        LOG.warning("HCU LightOp per-token INT8: skipped (lightop unavailable: %s)",
                    error)
        return False

    wrapper = _wrap(original)
    _BINDING = (module, wrapper, original)
    setattr(module, "scaled_int8_quant", wrapper)
    setattr(module, _MARKER, True)
    _arm_atexit()
    LOG.warning(
        "HCU adapter: LightOp per-token INT8 quantization installed on %s "
        "(dynamic per-token symmetric only; every other contract uses upstream)",
        _TARGET_SYMBOL,
    )
    return True