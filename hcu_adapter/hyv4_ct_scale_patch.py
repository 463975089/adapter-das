# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU adapter: HYV4 weight loading keeps CT int8 per-channel ``.weight_scale``.

v0522 ``hunyuan_v4.py`` / ``hunyuan_v4_nextn.py`` load_weights rename every
checkpoint ``*.weight_scale`` to ``*.weight_scale_inv`` unconditionally (the
fp8 block-quant param name). Compressed-tensors W8A8Int8 / WNA16 register the
per-channel scale as ``weight_scale``, so the DAS Hy4 Channel-INT4-w4a8
checkpoint (int8 mla/mlp/indexer + int4 routed experts) dies at load with
``KeyError: 'model.layers.N.mlp.shared_experts.gate_up_proj.weight_scale_inv'``
(S1 2026-11-03). The DAS fork's fix (``hyv4_linear_scale_suffix``) picks the
suffix from the registered params: ``_inv`` only when the model actually owns
``*.weight_scale_inv`` params (fp8 block), else keep the name.

Platform-independent upstream gap (CUDA would KeyError identically). The
runtime bridge gives each HYV4 class a class-local ``do_load_weights``: when
the model registers no ``*.weight_scale_inv`` param it strips the ``_inv``
that the community ``mapped_weights`` appended, then delegates to the
inherited DeepseekV2WeightLoaderMixin loader. Community ``load_weights``, the
mixin function object and other mixin users (DeepseekV2/V3) are untouched;
no source reads, recompilation or file writes.
Gate: ``HCU_HYV4_CT_SCALE=0`` skips installation.
"""

import functools
import importlib
import inspect
import logging
import threading

LOG = logging.getLogger("hcu.adapter")

_LOCK = threading.RLock()
_MARKER = "_hcu_ct_scale_original"
_INV = ".weight_scale_inv"
_TARGETS = (
    ("sglang.srt.models.hunyuan_v4", "HYV4ForCausalLM"),
    ("sglang.srt.models.hunyuan_v4_nextn", "HYV4ForCausalLMNextN"),
)


def _keep_ct_scale_names(weights):
    for name, loaded_weight in weights:
        if name.endswith(_INV):
            name = name[: -len("_inv")]
        yield name, loaded_weight


def _install_class(cls):
    """Shadow the inherited do_load_weights on `cls` only (idempotent)."""
    with _LOCK:
        current = cls.__dict__.get("do_load_weights")
        if current is not None:
            if getattr(current, _MARKER, None) is None:
                raise RuntimeError(
                    "hyv4_ct_scale: %s defines its own do_load_weights;"
                    " upstream loader changed" % cls.__name__
                )
            return cls
        original = getattr(cls, "do_load_weights", None)
        if not callable(original):
            raise RuntimeError("hyv4_ct_scale: %s has no do_load_weights" % cls.__name__)
        signature = inspect.signature(original)
        parameters = signature.parameters
        if (list(parameters) != ["self", "weights", "is_nextn"]
                or parameters["is_nextn"].default is not False):
            raise RuntimeError("hyv4_ct_scale: upstream do_load_weights signature changed")

        @functools.wraps(original)
        def do_load_weights(self, weights, is_nextn=False):
            if not any(n.endswith(_INV) for n, _ in self.named_parameters()):
                weights = _keep_ct_scale_names(weights)
            return original(self, weights, is_nextn=is_nextn)

        do_load_weights.__signature__ = signature
        setattr(do_load_weights, _MARKER, original)
        cls.do_load_weights = do_load_weights
        return cls


def install():
    from . import hcu_env, registry

    if not hcu_env.enabled("HCU_HYV4_CT_SCALE"):
        return
    for module_name, class_name in _TARGETS:
        cls = getattr(importlib.import_module(module_name), class_name)
        _install_class(cls)
        registry.register(
            "platform_detect", "hyv4-ct-scale-name",
            target=f"{module_name}.{class_name}.do_load_weights",
            mechanism=registry.MECH_REBIND,
            detail="class-local do_load_weights keeps CT int8 .weight_scale when the"
                   " model has no *_inv params; community load_weights and mixin"
                   " retained; no source writes",
        )
    LOG.warning("HCU adapter: HYV4 CT-int8 .weight_scale naming adapted at runtime")
