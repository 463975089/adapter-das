# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""tvm-ffi torch dlpack 首编自递归 breaker（R-18 自动化；validation-host-26 2026-10-26）。

tvm_ffi 首次 import 时若缓存缺失会派一个 python 构建子进程
（``_optional_torch_c_dlpack.load_torch_c_dlpack_extension`` →
``subprocess.run``）。子进程继承 PYTHONPATH → 重跑本 adapter 的
sitecustomize → 桥的 eager import 链（dsa_topk_v2 的 _SCORING_MOD 一路
拖进 xgrammar → tvm_ffi）→ 缓存仍缺 → 再派构建子进程……形成父子自递归：
bootstrap.md §3 记录的 122 进程/50 分钟案；validation-host-26 dsv41f 冷容器 2026-10-26
复现（9+ python3 堆积，签名 = 每 ~30s 孵一个新进程、各烧 ~30s import 后
阻塞在 select 等自己的子进程）。

修法 = 把 bootstrap §3 的手工预热自动化到「任何桥第一次可能触到 tvm_ffi
之前」（sitecustomize 编排层 install，先于 dsa_topk_v2 等重桥）：

1. 探温快路径：``~/.cache/tvm-ffi/**.so`` 任一存在 → 直接返回（幂等秒过）；
2. 冷则 flock 排他（TP spawn 风暴下只放一个构建者，防 §3 标题的「JIT
   竞争死锁」），锁内复检后再动手；
3. 以 **剥掉 PYTHONPATH 的干净 env** 亲跑 tvm_ffi 自带构建脚本
   （``utils/_build_optional_torch_c_dlpack.py``，参数照 §3 配方）——
   子进程无 PYTHONPATH → sitecustomize 不加载 → 单层走完落缓存；
4. 兜底：若 helper 缺席/失败，退化为窗口内 ``import tvm_ffi`` 让加载器
   自己派构建子进程，``_CleanEnvPopen`` 窗口给每个子进程注入同款干净
   env（递归物理断路，不依赖脚本名/libname 版本对齐），窗口结束还原。

失败路径：构建失败只告警一次并放行，tvm_ffi 稍后以自身语义响亮报错；
tvm_ffi 未安装则静默跳过。调试层：``HCU_TVMFFI_WARMUP=0`` 关闭。
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("hcu.adapter")

_LOCK_NAME = ".hcu_dlpack_build.lock"

_registered = False


def _cache_dir():
    """tvm-ffi dlpack 缓存目录（bootstrap §3 契约位：~/.cache/tvm-ffi）。"""
    return Path.home() / ".cache" / "tvm-ffi"


def _cache_so(cache_dir):
    """缓存里已有的 .so 列表；非空即视为暖（§3 验收口径：出现 .so 即成）。"""
    try:
        return list(cache_dir.rglob("*.so"))
    except OSError:
        return []


def _torch_tag():
    import torch

    maj, mini = torch.__version__.split(".")[:2]
    return f"torch{maj}{mini}"


def _helper_script():
    """tvm_ffi 自带构建脚本路径；find_spec 只定位不执行（执行即触发加载链）。"""
    import importlib.util

    try:
        spec = importlib.util.find_spec("tvm_ffi")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    helper = Path(spec.origin).parent / "utils" / "_build_optional_torch_c_dlpack.py"
    return helper if helper.exists() else None


def _build_with_clean_env(helper, cache_dir):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [
        sys.executable,
        str(helper),
        "--output-dir",
        str(cache_dir),
        "--libname",
        f"libtorch_c_dlpack_addon_{_torch_tag()}-rocm.so",
        "--build-with-rocm",
    ]
    LOG.warning(
        "HCU adapter: tvm-ffi dlpack cache cold -> one-time build "
        "(~20-40s; recursion-safe env, no PYTHONPATH)"
    )
    subprocess.run(cmd, env=env, check=False, capture_output=True, timeout=900)


class _CleanEnvPopen:
    """窗口内包装 subprocess.Popen：env 缺省的子进程一律注入剥 PYTHONPATH
    的干净 env。仅覆盖预热窗口（sitecustomize 单线程期），进窗口即装、
    出窗口即还原——tvm_ffi 加载器自派的构建子进程由此断掉重进
    sitecustomize 的路，不依赖我们拼对 helper 的参数。"""

    def __init__(self):
        self._orig = None

    def __enter__(self):
        self._orig = subprocess.Popen

        def patched(*args, **kwargs):
            if kwargs.get("env") is None:
                kwargs["env"] = {
                    k: v for k, v in os.environ.items() if k != "PYTHONPATH"
                }
            return self._orig(*args, **kwargs)

        subprocess.Popen = patched
        return self

    def __exit__(self, *exc):
        subprocess.Popen = self._orig
        return False


def install():
    """冷缓存则预热；返回 True 表示本次做了动作（构建/兜底加载）。"""
    global _registered

    from hcu_adapter import hcu_env

    if not hcu_env.enabled("HCU_TVMFFI_WARMUP"):
        return False
    cache_dir = _cache_dir()
    if _cache_so(cache_dir):
        return False  # 暖缓存：秒过，不碰任何 import

    try:
        import fcntl
    except ImportError:
        fcntl = None  # 非 POSIX（开发机单测）：退化为无锁，语义不变

    acted = False
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if fcntl is not None:
            lock = open(cache_dir / _LOCK_NAME, "w")
            fcntl.flock(lock, fcntl.LOCK_EX)
        else:
            lock = None
        try:
            if _cache_so(cache_dir):  # 锁内复检：别的进程刚建完
                return False
            helper = _helper_script()
            built = False
            if helper is not None:
                acted = True
                try:
                    _build_with_clean_env(helper, cache_dir)
                    built = bool(_cache_so(cache_dir))
                except Exception:
                    LOG.warning(
                        "HCU adapter: tvm-ffi dlpack manual build failed",
                        exc_info=True,
                    )
            if not built:
                # 兜底：tvm_ffi 自己派构建子进程；窗口包装断递归
                with _CleanEnvPopen():
                    import tvm_ffi  # noqa: F401
                acted = True
                built = bool(_cache_so(cache_dir))
            if built:
                LOG.warning(
                    "HCU adapter: tvm-ffi dlpack cache warm (.so present); "
                    "HCU_TVMFFI_WARMUP=0 to opt out"
                )
            else:
                LOG.warning(
                    "HCU adapter: tvm-ffi dlpack cache still absent after "
                    "warm attempt — tvm_ffi will surface its own error later"
                )
        finally:
            if lock is not None:
                lock.close()  # close 即释放 flock
    except ImportError:
        LOG.warning(
            "HCU adapter: tvm-ffi warmup skipped (tvm_ffi not installed)"
        )
        return acted
    if acted and not _registered:
        _registered = True
        try:
            from hcu_adapter import registry

            registry.register(
                "platform_detect",
                "tvmffi-dlpack-warmup",
                target="env:~/.cache/tvm-ffi",
                mechanism=registry.MECH_REBIND,
                detail="cold cache -> clean-env build (R-18 recursion breaker)",
            )
        except Exception:  # catalog/governance drift: guard still acts
            pass
    return acted
