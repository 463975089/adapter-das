# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CT W4A8 MoE scheme 路由桥（validation-host-26 dsv41f Channel-INT4-w4a8，2026-10-26）。

上游事实（v0522 compressed_tensors.py，validation-host-26 实树核对）：

- ``target_scheme_map`` 按**每个 target 字符串**建键
  （``_quantization_scheme_map_from_config`` 遍历 config_groups 的 targets）。
  正典 llmcompressor 输出带 ``targets:["Linear"]`` 组 → 有 "Linear" 键；
  DAS 转换器（DeepSeek-V4.1-Flash-Channel-INT4-w4a8）全部用 regex targets
  → 键就是 regex 串本身，"Linear" 不存在。
- ``CompressedTensorsW4AFP8MoE.__init__``（w4a8_fp8_moe.py:87）硬读
  ``target_scheme_map["Linear"]["weights"]`` 取 num_bits/group_size/symmetric
  ——这些值与分支选择时已按 regex 匹配到的 ``weight_quant`` 完全相同，读
  "Linear" 纯属对正典布局的假设 → KeyError，模型构建期死（权重未加载）。
- 补上 KeyError 仍不足以执行：该 checkpoint 的 channel group_size=-1
  会使 stock create_weights 的尺度维度为负；stock 标准路径实际使用
  tensor 级 FP8 激活及固定 group_size=128 的 CUTLASS GEMM，不能等同于
  checkpoint 声明的 token 级 INT8 激活。参数与内核契约均需另行验证。

修法（governance 判别问句：上游 CUDA 有 W4A8 MoE 行为 → 行为还原归
adapter；nvfp4 w4a4 capability<10 时上游自己也降级 w4a16"activation
quantization is ignored"，本桥 w4a16 路由沿用同一模式）：

- ``HCU_W4A8_MOE=w4a16``（默认）：wrap ``CompressedTensorsConfig.__init__``，
  构造后把 target_scheme_map 中 **int4 权重组**的 ``input_activations``
  置 None（dense int8/int8-act 组不动）。上游分支链随即自然走
  ``_is_wNa16_group_channel(w, None)`` 的 ``_is_hip`` 臂 →
  ``CompressedTensorsWNA16TritonMoE``（ROCm），即 box16 dsv41f-INT4 已
  验证路径，再由 10.6 channel 改写桥（HCU_WNA16_CHANNEL_MOE）收口
  block_shape。激活量化被移除 → MoE 激活跑 bf16；这是不同的数值路径，
  不能据此宣称与原 W4A8 等价或整模型精度更高。
- ``HCU_W4A8_MOE=heal``：wrap ``get_moe_scheme``，捕获 KeyError('Linear')
  后**作用域内**临时注入 "Linear" 桶（内容 = 该层 regex 匹配到的
  scheme_dict，即上游本来就要用的那份），重试原函数，用毕即删——dense
  int8 组不受污染。保留 stock W4AFP8 scheme 的选择，后续仍需验证
  channel/group 参数、实际激活量化契约及 HIP 内核；修复构造期 KeyError
  本身不代表该路径可以执行。
- ``HCU_W4A8_MOE=0``：全关，还原 stock KeyError。
- ``HCU_W4A8_MOE=int8``：显式候选，保留激活配置，复用社区 raw INT4
  加载器，接 BF16 DeepEP low_latency 的原 dtype 整数专家合同；
  仅支持已验证的对称 channel/token 组和简单 SiLU 配置，未验收整模型精度。
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
    if route in ("", "0", "None", "refused") or _installed:
        return

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
        if route != "w4a16":
            LOG.warning(
                "HCU adapter: HCU_W4A8_MOE=%r unrecognized, using w4a16", route
            )
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
