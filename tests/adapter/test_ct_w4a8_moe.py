# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""ct_w4a8_moe 路由桥契约（CPU 单测，任何机器；sglang 树不需要在场）。

契约面：
1. _null_int4_act_quant：只动 int4 权重组的 input_activations（int8 组、
   fp8-act int4 组、已 None 组的行为各归各）；
2. w4a16 路由 patched_init：原构造语义透传 + 变换幂等；
3. heal 路由 wrapped get_moe_scheme：KeyError('Linear') 时按该层
   regex 匹配到的 int4 组临时注入 "Linear" 桶、重试、用毕即删；
   非 'Linear' KeyError 不吞；int8 层不 heal；已有 "Linear" 直走原函数。
"""

from types import SimpleNamespace as NS

from hcu_adapter import ct_w4a8_moe as cw


def _args(type_name, num_bits, **kw):
    return NS(type=NS(name=type_name), num_bits=num_bits, **kw)


def _args_strtype(type_str, num_bits, **kw):
    """pydantic use_enum_values 形态：type 是字符串（nmz16 探针实锤）。"""
    return NS(type=type_str, num_bits=num_bits, **kw)


def _entry(w=None, a=None, fmt=None):
    return {"weights": w, "input_activations": a, "format": fmt}


W4 = _args("INT", 4, group_size=-1, symmetric=True)
W4S = _args_strtype("int", 4, group_size=-1, symmetric=True)  # pydantic 字符串形态
W8 = _args("INT", 8)
ACT8 = _args("INT", 8, dynamic=True, strategy="token")
ACT8F = _args("FLOAT", 8, dynamic=True)


def test_null_only_int4_groups():
    m = {
        "re:routed": _entry(W4, ACT8, "pack-quantized"),
        "re:mtp": _entry(W4S, ACT8, "pack-quantized"),
        "re:attn": _entry(W8, ACT8, "int-quantized"),
        "re:shared": _entry(W8, ACT8F, "int-quantized"),
        "re:w4a16": _entry(W4, None, "pack-quantized"),
    }
    n = cw._null_int4_act_quant(m)
    assert n == 2
    assert m["re:routed"]["input_activations"] is None
    assert m["re:mtp"]["input_activations"] is None
    # dense int8 组（无论 act int/fp8）不动
    assert m["re:attn"]["input_activations"] is ACT8
    assert m["re:shared"]["input_activations"] is ACT8F
    # 已 None 幂等
    assert cw._null_int4_act_quant(m) == 0


def test_patched_init_passthrough_and_transform():
    state = {}

    def fake_init(self, a, b=None):
        state["args"] = (a, b)
        self.target_scheme_map = a

    patched = cw._make_patched_init(fake_init)
    obj = NS()
    m = {"re:routed": _entry(W4, ACT8)}
    patched(obj, m, b=7)
    assert state["args"] == (m, 7)  # 原构造语义透传
    assert m["re:routed"]["input_activations"] is None  # 变换生效


def test_heal_scoped_linear_injection():
    calls = {"orig": 0, "scheme": 0}

    def orig(self, layer, layer_name=None):
        calls["orig"] += 1
        if "Linear" not in self.target_scheme_map:
            raise KeyError("Linear")
        return "scheme-ok"

    class FakeCfg:
        def __init__(self):
            self.target_scheme_map = {"re:routed": _entry(W4, ACT8)}

        def get_scheme_dict(self, layer, name):
            calls["scheme"] += 1
            return self.target_scheme_map["re:routed"]

    wrapped = cw._make_healed_get_moe_scheme(orig)
    cfg = FakeCfg()
    assert wrapped(cfg, "layer", layer_name="model.layers.3.ffn") == "scheme-ok"
    assert calls == {"orig": 2, "scheme": 1}
    assert "Linear" not in cfg.target_scheme_map  # 用毕即删


def test_heal_rejects_non_int4_and_foreign_keyerror():
    def orig(self, layer, layer_name=None):
        raise KeyError("Linear")

    class Int8Cfg:
        target_scheme_map = {"re:attn": _entry(W8, ACT8)}

        def get_scheme_dict(self, layer, name):
            return self.target_scheme_map["re:attn"]

    wrapped = cw._make_healed_get_moe_scheme(orig)
    import pytest

    with pytest.raises(KeyError):
        wrapped(Int8Cfg(), "layer", layer_name="x")

    def orig_other(self, layer, layer_name=None):
        raise KeyError("other")

    wrapped2 = cw._make_healed_get_moe_scheme(orig_other)
    with pytest.raises(KeyError):
        wrapped2(Int8Cfg(), "layer", layer_name="x")


def test_heal_shortcuts_when_linear_present():
    def orig(self, layer, layer_name=None):
        return "canonical"

    class CanonicalCfg:
        target_scheme_map = {"Linear": _entry(W8, ACT8)}

    wrapped = cw._make_healed_get_moe_scheme(orig)
    assert wrapped(CanonicalCfg(), "l", layer_name="x") == "canonical"


def test_type_name_tolerance():
    assert cw._type_name(W4) == "INT"
    assert cw._type_name(W4S) == "INT"  # 字符串 "int" 归一
    assert cw._type_name(None) == ""
    assert not cw._is_int4_weights(None)
    assert cw._is_int4_weights(W4)
    assert cw._is_int4_weights(W4S)  # nmz16 探针实锤的形态
    assert not cw._is_int4_weights(W8)
