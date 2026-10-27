"""dsv4_wqkv_scale_patch 针脚契约（CPU 单测，任何机器，无需 sglang 树）。

契约面：
1. _apply：needle 命中 → 注入 wq_a/wkv 两条 .weight_scale endswith；
2. 幂等：重复 apply 不二次注入；
3. 自愈：marker 在而 needle 失踪（上游合入修复/重构）→ 原样返回不炸；
4. needle 真漂移 → RuntimeError（governance §3）；
5. 门关（HCU_DSV4_WQKV_SCALE=0）→ install 在碰 sglang 之前返回。
"""

import pytest

from sgl_kernel._adapters import dsv4_wqkv_scale_patch as dp


def _snippet():
    return (
        "                            elif (\n"
        "                                fuse_wqa_wkv\n"
        "                                and (\n"
        '                                    name.endswith(".wq_a.weight")\n'
        '                                    or name.endswith(".wq_a.weight_scale_inv")\n'
        '                                    or name.endswith(".wkv.weight")\n'
        '                                    or name.endswith(".wkv.weight_scale_inv")\n'
        '                                    or name.endswith(".wq_a.qweight")\n'
        "                                )\n"
        "                            ):\n"
    )


def test_apply_injects_both_scale_arms():
    out = dp._apply(_snippet())
    assert 'name.endswith(".wq_a.weight_scale")' in out
    assert 'name.endswith(".wkv.weight_scale")' in out
    # 原有三行保持不动（inv/wkv.weight 仍在）
    assert 'name.endswith(".wq_a.weight_scale_inv")' in out
    assert 'name.endswith(".wkv.weight")' in out


def test_apply_idempotent():
    once = dp._apply(_snippet())
    twice = dp._apply(once)
    assert once == twice
    assert once.count('name.endswith(".wq_a.weight_scale")') == 1


def test_marker_selfheal_when_needle_gone():
    healed = _snippet().replace(
        'name.endswith(".wq_a.weight_scale_inv")',
        'name.endswith(".wq_a.weight_scale")  # HCU dsv4_wqkv_scale: x',
    )
    assert dp._apply(healed) == healed  # marker 在：上游已修/重构，静默放行


def test_needle_drift_raises():
    with pytest.raises(RuntimeError, match="needle missing"):
        dp._apply("def load_weights():\n    pass\n")


def test_gate_off_returns_before_sglang(monkeypatch):
    from sgl_kernel._adapters import hcu_env

    monkeypatch.setattr(hcu_env, "enabled", lambda name: False)
    assert dp.install() is None  # 未触 sglang import（本机无树也不炸）
