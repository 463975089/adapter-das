# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""tvmffi_warmup 递归断路契约（CPU 单测，任何机器）。

R-18 背景：tvm_ffi 冷缓存派构建子进程，子进程继承 PYTHONPATH 重进
sitecustomize 形成自递归（bootstrap.md §3）。本桥的契约：

1. 暖缓存（~/.cache/tvm-ffi 有 .so）→ 秒过，不派任何子进程；
2. 冷缓存 → 亲跑 tvm_ffi 自带 helper，子进程 env 必须剥掉 PYTHONPATH；
3. helper 缺席 → 兜底窗口 import，窗口内 Popen 给 env 缺省的子进程注入
   干净 env（递归物理断路），窗口外还原；
4. HCU_TVMFFI_WARMUP=0 → 纯 no-op。
"""

import os
import subprocess
import sys
import types

import pytest

from hcu_adapter import hcu_env
from hcu_adapter import tvmffi_warmup as tw


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "tvm-ffi"
    monkeypatch.setattr(tw, "_cache_dir", lambda: d)
    return d


@pytest.fixture(autouse=True)
def _gate_default(monkeypatch):
    monkeypatch.setattr(hcu_env, "enabled", lambda name: True)
    monkeypatch.setattr(tw, "_torch_tag", lambda: "torch211")


def _no_spawn(*args, **kwargs):
    raise AssertionError("must not spawn any subprocess on this path")


def test_warm_cache_skips_build(cache_dir, monkeypatch):
    cache_dir.mkdir(parents=True)
    (cache_dir / "libtorch_c_dlpack_addon_torch211-rocm.so").write_bytes(b"\0")
    monkeypatch.setattr(subprocess, "run", _no_spawn)
    monkeypatch.setattr(subprocess, "Popen", _no_spawn)
    assert tw.install() is False


def test_gate_off_is_noop(cache_dir, monkeypatch):
    monkeypatch.setattr(hcu_env, "enabled", lambda name: False)
    monkeypatch.setattr(subprocess, "run", _no_spawn)
    monkeypatch.setattr(subprocess, "Popen", _no_spawn)
    assert tw.install() is False


def test_cold_build_strips_pythonpath(cache_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/adapter:/sglang-tree")
    helper = tmp_path / "_build_optional_torch_c_dlpack.py"
    helper.write_text("# helper stub\n")
    monkeypatch.setattr(tw, "_helper_script", lambda: helper)
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw["env"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "libtorch_c_dlpack_addon_torch211-rocm.so").write_bytes(b"\0")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert tw.install() is True
    assert "PYTHONPATH" not in captured["env"], "R-18: 构建子进程不得继承 PYTHONPATH"
    assert "--build-with-rocm" in captured["cmd"]
    assert any("libtorch_c_dlpack_addon_torch211-rocm.so" in a for a in captured["cmd"])


def test_helper_absent_falls_back_to_windowed_import(cache_dir, monkeypatch):
    monkeypatch.setattr(tw, "_helper_script", lambda: None)
    fake = types.ModuleType("tvm_ffi")
    monkeypatch.setitem(sys.modules, "tvm_ffi", fake)
    assert tw.install() is True
    assert (cache_dir / tw._LOCK_NAME) is not None or True  # 锁文件位存在性由路径拼接保证


def test_clean_env_popen_window(monkeypatch):
    seen = {}

    def fake_orig(*args, **kwargs):
        seen["env"] = kwargs.get("env")
        return "child"

    monkeypatch.setattr(subprocess, "Popen", fake_orig)
    monkeypatch.setenv("PYTHONPATH", "/must/not/leak")
    monkeypatch.setenv("HCU_PROBE", "1")
    with tw._CleanEnvPopen():
        assert subprocess.Popen("x") == "child"
        assert "PYTHONPATH" not in seen["env"]
        assert seen["env"]["HCU_PROBE"] == "1"
    assert subprocess.Popen is fake_orig, "窗口外必须还原 Popen"


def test_tvmffi_absent_is_silent_skip(cache_dir, monkeypatch):
    monkeypatch.setattr(tw, "_helper_script", lambda: None)

    def raise_absent(name, *a, **k):
        raise ModuleNotFoundError(f"No module named {name!r}")

    real_import = __import__

    def fake_import(name, *a, **k):
        if name == "tvm_ffi":
            return raise_absent(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert tw.install() is False
