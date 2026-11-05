# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CUDA 版本探针双命名空间替换的纯逻辑 pin（2026-11-05）。

事故：sglang main 的 kernels 模块在 import 时经 ``sglang.srt.utils`` 包级
re-export（``from .common import *`` 冻结的原 lru_cache 对象）调用版本探针
（fused_sigmoid_gating_recurrent ``_is_sm90 = is_sm90_supported()``，kda 后端
链），原函数体末尾 ``torch.version.cuda.split`` 在 HCU（cuda=None）上直接
AttributeError——bw47 hybrid/eager 实炸。platform_seam 旧版只 patch
``utils.common`` 命名空间，包级拷贝漏网。

本文件用假模块按上游真实形态（lru_cache(partial(_check_cuda_device_version,
...))，__wrapped__ 指回 partial）钉住 _replace_version_gates 的行为：
两个命名空间同步替换、替换对象共享、非探针符号零误伤。

Run: python tests/adapter/test_platform_version_gates.py  (或 pytest)
"""

import functools
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter import platform_seam  # noqa: E402


def _fake_common():
    """Mirror sglang main utils/common shapes (no sglang import needed)."""

    def _check_cuda_device_version(majors, version):
        return True

    common = types.ModuleType("fake_utils_common")
    common._check_cuda_device_version = _check_cuda_device_version
    common.is_sm90_supported = functools.lru_cache(maxsize=1)(
        functools.partial(_check_cuda_device_version, majors=[9])
    )
    common.is_ampere_with_cuda_12_3 = functools.lru_cache(maxsize=1)(
        functools.partial(_check_cuda_device_version, majors=[8])
    )
    # non-gate symbols the loop must not touch
    common.get_device_sm = lambda: 936
    common.is_cuda = lambda: True
    common._check_cuda_device_version = _check_cuda_device_version  # itself
    return common


def _frozen_pkg(common):
    """Package layer: `from .common import *` frozen copies (the bug)."""
    pkg = types.ModuleType("fake_utils")
    pkg.is_sm90_supported = common.is_sm90_supported
    pkg.is_ampere_with_cuda_12_3 = common.is_ampere_with_cuda_12_3
    pkg.get_device_sm = common.get_device_sm
    pkg.is_cuda = common.is_cuda
    return pkg


def test_version_gates_replaced_in_both_namespaces():
    common = _fake_common()
    pkg = _frozen_pkg(common)
    gates = platform_seam._replace_version_gates(common, pkg)

    assert set(gates) == {"is_sm90_supported", "is_ampere_with_cuda_12_3"}
    for obj in (common, pkg):
        assert obj.is_sm90_supported() is False
        assert obj.is_ampere_with_cuda_12_3() is False
        # untouched neighbors
        assert obj.get_device_sm() == 936
        assert obj.is_cuda() is True
    # same replacement object in both namespaces (identity, not just value)
    assert common.is_sm90_supported is pkg.is_sm90_supported


def test_pkg_none_patches_common_only():
    common = _fake_common()
    gates = platform_seam._replace_version_gates(common, None)
    assert common.is_sm90_supported() is False
    assert set(gates) == {"is_sm90_supported", "is_ampere_with_cuda_12_3"}


def test_missing_check_is_a_noop():
    common = types.ModuleType("bare")
    common.is_sm90_supported = lambda: True
    assert platform_seam._replace_version_gates(common, None) == []
    assert common.is_sm90_supported() is True


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
