# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Optional compatibility routes for compressed-tensors MoE.

The default native route preserves the checkpoint's activation quantization.
It does not supply a missing W4A8 kernel: unsupported schemes fail upstream.
Legacy W4A16 conversion is available only when explicitly requested and must
not be used for W4A8 validation. Dense INT8 groups are never converted.
The existing explicit int8 candidate remains available with its own contracts.
"""

import logging

LOG = logging.getLogger("hcu.adapter")

_installed = False


def _type_name(obj) -> str:
    """QuantizationArgs.type 的归一：pydantic use_enum_values 下是字符串
    （"int"/"float"），原生枚举下是 QuantizationType（.name 大写）。两种
    形态都归一到大写字符串（validation-host-16 探针实锤：字符串形态，首版假设枚举
    导致变换空转、KeyError 复发）。"""
    t = getattr(obj, "type", None)
    if t is None:
        return ""
    name = getattr(t, "name", t)
    return str(name).upper()


def _is_int4_weights(weights) -> bool:
    """QuantizationArgs 形状的 int4 判定（type=INT + num_bits 4）。"""
    return weights is not None and _type_name(weights) == "INT" and getattr(
        weights, "num_bits", None
    ) == 4


def _null_int4_act_quant(target_scheme_map) -> int:
    """w4a16 路由的数据变换：int4 权重组的 input_activations 置 None。

    只动 int4 条目（routed/MTP 专家、canonical W4AFP8 的 fp8-act 同样
    覆盖）；dense int8 组（weights num_bits=8）原样。返回改动条数。
    """
    n = 0
    for entry in target_scheme_map.values():
        if not isinstance(entry, dict):
            continue
        if _is_int4_weights(entry.get("weights")) and entry.get(
            "input_activations"
        ) is not None:
            entry["input_activations"] = None
            n += 1
    return n


def _make_patched_init(orig_init):
    """w4a16 路由：包装 CompressedTensorsConfig.__init__。

    原始构造完成后做一次数据变换（幂等：二次构造同 config 仍正确，
    因为 from_config 每次重新 parse）。
    """

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            n = _null_int4_act_quant(self.target_scheme_map)
            if n:
                LOG.warning(
                    "HCU adapter: CT W4A8 -> W4A16 route: dropped "
                    "input_activations on %d int4 group(s); MoE activations "
                    "stay bf16 (upstream WNA16TritonMoE ROCm arm); "
                    "HCU_W4A8_MOE=heal for stock-scheme pass-through",
                    n,
                )
        except Exception:
            LOG.warning(
                "HCU adapter: CT W4A8 -> W4A16 route transform failed",
                exc_info=True,
            )

    return patched_init


def _make_healed_get_moe_scheme(orig):
    """heal 路由：包装 get_moe_scheme，作用域内补 KeyError('Linear')。

    临时 "Linear" 桶取该层自己 regex 匹配到的 scheme_dict（上游 branch
    本来就基于它选的 scheme，值一致）；重试毕即删，绝不过夜——防 "Linear"
    作为模块名 target 参与后续 get_scheme_dict 匹配、劫持 dense int8 组。
    """

    def get_moe_scheme(self, layer, layer_name=None):
        if "Linear" in self.target_scheme_map:
            return orig(self, layer, layer_name=layer_name)
        try:
            return orig(self, layer, layer_name=layer_name)
        except KeyError as exc:
            if not exc.args or exc.args[0] != "Linear":
                raise
            scheme_dict = None
            base = layer_name or ""
            for proj in (".0.gate_proj", ".0.up_proj", ".0.down_proj"):
                try:
                    cand = self.get_scheme_dict(layer, base + proj)
                except Exception:
                    cand = None
                if cand is not None:
                    scheme_dict = cand
                    break
            if scheme_dict is None or not _is_int4_weights(
                scheme_dict.get("weights")
            ):
                raise
            LOG.warning(
                "HCU adapter: CT W4A8 scheme KeyError('Linear') healed "
                "from the regex-matched group (%s); stock W4AFP8 scheme "
                "selected; channel/group parameters, activation quantization "
                "and HIP kernels still require validation",
                layer_name,
            )
            self.target_scheme_map["Linear"] = {
                "weights": scheme_dict.get("weights"),
                "input_activations": scheme_dict.get("input_activations"),
                "format": scheme_dict.get("format"),
            }
            try:
                return orig(self, layer, layer_name=layer_name)
            finally:
                self.target_scheme_map.pop("Linear", None)

    return get_moe_scheme


def install():
    global _installed

    from hcu_adapter import hcu_env

    route, _src = hcu_env.resolve("HCU_W4A8_MOE")
    if route in ("native", "", "0", "None", "refused") or _installed:
        return

    if route not in ("w4a16", "heal", "int8"):
        raise ValueError(
            f"Unsupported HCU_W4A8_MOE={route!r}; expected native, 0, w4a16, heal or int8"
        )

    from sglang.srt.layers.quantization.compressed_tensors import (
        compressed_tensors as _ct,
    )

    cls = _ct.CompressedTensorsConfig
    if route == "heal":
        cls.get_moe_scheme = _make_healed_get_moe_scheme(cls.get_moe_scheme)
        detail = "get_moe_scheme KeyError('Linear') scoped heal"
    elif route == "int8":
        from hcu_adapter import ct_w4a8_int8_moe

        ct_w4a8_int8_moe.install()
        detail = (
            "explicit channel INT4/token INT8 BF16 DeepEP LL candidate; "
            "input_activations retained; model accuracy not accepted"
        )
    else:
        cls.__init__ = _make_patched_init(cls.__init__)
        detail = (
            "__init__ drops int4-group input_activations -> WNA16TritonMoE"
            " (ROCm) route"
        )
        # The normal WNA16 route is enough for TP/EP1.  DeepEP low-latency
        # dispatch uses expert-major masked tensors and bypasses the regular
        # fused_experts seam, so install its adapter-owned W4A16 branch too.
        try:
            from hcu_adapter import w4a16_moe

            w4a16_moe.install_deepep_w4a16()
            detail += "; DeepEP LL expert-major W4A16 branch"
        except ImportError:
            pass
        except Exception:
            LOG.warning(
                "HCU adapter: DeepEP LL W4A16 branch not installed",
                exc_info=True,
            )
    _installed = True
    LOG.warning("HCU adapter: CT W4A8 MoE bridge installed (%s)", detail)

    try:
        from hcu_adapter import registry

        registry.register(
            "platform_detect",
            "ct-w4a8-moe",
            target="class:CompressedTensorsConfig",
            mechanism=registry.MECH_REBIND,
            detail=detail,
        )
    except Exception:  # catalog/governance drift: bridge still acts
        pass
