"""HCU adapter: gate DSV2/V4 capture-mode dual-stream MoE off on real HIP.

``DeepseekV2MoE.forward`` selects ``forward_normal_dual_stream`` ONLY under
``get_is_capture_mode()`` (the shared expert overlaps the routed experts on
``alt_stream``). On DTK/ROCm (gfx938) the cross-stream captured graph replays
with a broken shared-expert dependency: every replayed decode token becomes
deterministic garbage ('#####') while eager (single-stream
``forward_normal``) stays perfect. Eager is unaffected by this gate -- the
dual-stream branch is capture-only by construction.

The gate must test the true device fact (``torch.version.hip``): the
platform_seam (governance §0.5) flips sglang's ``is_hip()``/``_is_hip`` to
False in the server process, so an ``_is_hip``-based gate never fires.
Located: DSV4-Flash nmz16 2026-10-21 (fingerprint bisect: layer-0 post-attn
boundary all bit-exact; MoE in-place write-back showed the corruption;
moepath markers proved the captured path; full account in docs/models.md).

The bridge is now function-scoped runtime dispatch, not a Python source-file
patch. It is installed after operator bridges are ready, before model startup.
"""

import logging
import dis
import inspect
import threading
import types

LOG = logging.getLogger("hcu.adapter")

_LOCK = threading.RLock()
_TARGET = "_hcu_single_stream_moe_forward"
_MARKER = "_hcu_single_stream_moe_original"


def _not_dual_stream_capture():
    return False


def _forward(*args, **kwargs):
    return _hcu_single_stream_moe_forward(*args, **kwargs)


def _install_forward(function):
    """Suppress only the legacy dual-stream dispatch in this function's scope.

    Other functions keep the real capture flag, so this neither ends capture
    nor changes shared process state. The original forward still selects
    normal/DeepEP/mega-MoE and forwards skip_shared_experts itself.
    """
    with _LOCK:
        if not isinstance(function, types.FunctionType) or function.__closure__:
            raise RuntimeError("moe_dualstream_gate: expected closure-free forward")
        if hasattr(function, _MARKER):
            if (function.__code__ is not _forward.__code__
                    or function.__globals__.get(_TARGET) is not getattr(function, _MARKER)):
                raise RuntimeError("moe_dualstream_gate: forward changed after installation")
            return function
        references = [
            item for item in dis.get_instructions(function)
            if item.opname == "LOAD_GLOBAL" and item.argval == "get_is_capture_mode"
        ]
        if len(references) != 1 or not {
            "forward_normal", "forward_normal_dual_stream",
        }.issubset(function.__code__.co_names):
            raise RuntimeError("moe_dualstream_gate: upstream capture dispatch changed")
        if not callable(function.__globals__.get("get_is_capture_mode")):
            raise RuntimeError("moe_dualstream_gate: capture predicate is unavailable")
        if _TARGET in function.__globals__:
            raise RuntimeError("moe_dualstream_gate: private dispatch binding already exists")
        signature = inspect.signature(function)
        namespace = dict(
            function.__globals__, get_is_capture_mode=_not_dual_stream_capture,
        )
        original = types.FunctionType(
            function.__code__, namespace, function.__name__,
            function.__defaults__, function.__closure__,
        )
        original.__kwdefaults__ = function.__kwdefaults__
        function.__globals__[_TARGET] = original
        function.__signature__ = signature
        setattr(function, _MARKER, original)
        function.__code__ = _forward.__code__
        return function


def install():
    import torch

    if torch.version.hip is None:
        return  # hardware truth; do not change CUDA hosts or probe GPU memory
    import sglang.srt.models.deepseek_v2 as model
    import hcu_adapter.registry as _reg

    _install_forward(model.DeepseekV2MoE.forward)
    _reg.register(
        "platform_detect",
        "single-stream-moe-capture",
        target="sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward",
        mechanism=_reg.MECH_REBIND,
        detail="private dispatch capture predicate; upstream source and global capture state unchanged",
    )
    LOG.warning(
        "HCU adapter: DSV2 capture-mode dual-stream MoE gated at runtime on HIP"
        " (cross-stream graph replay corruption, gfx938)"
    )
