"""R-12: automatic bootstrap install-sequence regression pins（总装入口回归防护）.

三行 sitecustomize 自动调用 hcu_adapter.bootstrap，后者先装 common 守卫，
再选择唯一宿主桥；既有三十余座 sglang 桥的顺序链位于
hcu_adapter/bridges/sglang.py。本文件把该高爆炸半径启动面的五层覆盖钉成
可执行测试（2026-10-19，2026-10-27 随多宿主迁移更新）：

  L1 structure   AST 顺序/门控/隔离契约 + 门控 env 与 docs/env-vars.md 对账
                 （多一漏一即红；纯静态，任何机器可跑）
  L2 no-op       非 HIP 解释器下零副作用（子进程验证——no-op 本身是契约，
                 开发机/无卡 CI 可跑）
  L3 smoke       HIP 容器：干净子进程 import 后逐项断言落位（platform 语义
                 翻转 / rms_norm 重绑 / VLLM_PLUGINS setdefault）
  L4 gated arms  显式 env 永远赢 setdefault（HCU_MQA_LOGITS_IMPL / VLLM_PLUGINS
                 双臂）
  L5 instrument  HCU_ADAPTER_DUMP 布线：import 期 log_summary 落盘
                 registry.dump（ops/unbound/env 三键 = R-01 起验收标准通道）

头号方法学坑（PM 钉案）：pytest 进程自身已被 sitecustomize 自动 import 污染
——行为层一律走受控 env 的干净子进程，绝不停留在进程内状态。

Run: python tests/adapter/test_sitecustomize_sequence.py  （或 pytest；
ae0698b 起 L1 kernels 段收集 tests/adapter，本文件随段进 CI）
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from unittest import mock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SC_PATH = os.path.join(ROOT, "sitecustomize.py")
SGLANG_PATH = os.path.join(ROOT, "hcu_adapter", "bridges", "sglang.py")

IS_HIP = getattr(getattr(__import__("torch", fromlist=["version"]), "version"),
                 "hip", None) is not None

# 门控 env 读取的提取模式（routing glue 的裸 os.environ 读点；经
# hcu_env.resolve/enabled 的读取不在此列——那本就是正典路径）
_ENV_CALL = re.compile(
    r'os\.environ(?:\.get|\.setdefault)?\(\s*["\']([A-Z][A-Z0-9_]+)["\']')
_ENV_SUBSCRIPT = re.compile(
    r'os\.environ\[\s*["\']([A-Z][A-Z0-9_]+)["\']\s*\]')


# ---------------------------------------------------------------- L1 结构层

def _load_ast(path=SGLANG_PATH):
    src = open(path, encoding="utf-8").read()
    return src, ast.parse(src)


def _find_hip_gate(tree, src):
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            seg = ast.get_source_segment(src, node.test) or ""
            if "torch.version" in seg and "hip" in seg:
                return node
    return None


def _call_name(call):
    f = call.func
    if isinstance(f, ast.Attribute):
        base = f.value.id if isinstance(f.value, ast.Name) else ""
        return base, f.attr
    return "", ""


def _parent_map(tree):
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def test_structure_gate_order_and_isolation():
    """顺序契约：hip 门包裹全链；VLLM_PLUGINS setdefault 先于一切；
    platform_seam.install 必须是宿主链首个触 sglang 的步骤；每桥 install
    包 try/except（单桥失败不拖垮其余）。common-before-host 由 bootstrap
    契约测试单独固定。"""
    src, tree = _load_ast()
    gate = _find_hip_gate(tree, src)
    assert gate is not None, "hip gate `if torch.version.hip ... is not None` missing"

    # gate 的首条语句 = VLLM_PLUGINS setdefault（非空 allowlist 语义）
    first = gate.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), (
        "gate.body[0] should be the VLLM_PLUGINS setdefault call")
    base, attr = _call_name(first.value)
    assert attr == "setdefault", f"gate.body[0] is .{attr}(), expected setdefault"
    assert getattr(first.value.args[0], "value", None) == "VLLM_PLUGINS"

    def stmt_index(match):
        for i, st in enumerate(gate.body):
            for n in ast.walk(st):
                if isinstance(n, ast.Call):
                    b, a = _call_name(n)
                    if match(b, a):
                        return i
        return -1

    i_platform = stmt_index(lambda b, a: a == "install" and b == "platform_seam")
    assert i_platform >= 0, "platform_seam.install() call not found"

    # platform_seam.install 之前不得 import sglang（common.py import 期冻结
    # FP8_E4M3_MAX，seam 必须最先翻转语义）
    platform_lineno = None
    for n in ast.walk(gate.body[i_platform]):
        if isinstance(n, ast.Call):
            b, a = _call_name(n)
            if a == "install" and b == "platform_seam":
                platform_lineno = n.lineno
                break
    before = src.splitlines()[: platform_lineno - 1]
    offenders = [ln.strip() for ln in before
                 if re.match(r"(from sglang|import sglang)\b", ln.strip())]
    assert not offenders, f"sglang imported before platform_seam.install: {offenders}"

    # 每座桥的 install/install_patch 都被 try 包裹（异常隔离契约）
    parents = _parent_map(tree)
    naked = []
    for n in ast.walk(gate):
        if isinstance(n, ast.Call):
            b, a = _call_name(n)
            if a in ("install", "install_patch"):
                p, wrapped = parents.get(n), False
                while p is not None and p is not gate:
                    if isinstance(p, ast.Try):
                        wrapped = True
                        break
                    p = parents.get(p)
                if not wrapped:
                    naked.append(f"{b}.{a}@line{n.lineno}")
    assert not naked, f"install calls outside any try/except: {naked}"


def test_sitecustomize_delegates_to_bootstrap_only():
    """自动入口不得重新长回宿主实现体。"""
    src, tree = _load_ast(SC_PATH)
    executable = [node for node in tree.body if not isinstance(node, ast.Expr)
                  or not isinstance(getattr(node, "value", None), ast.Constant)]
    assert len(executable) == 2
    imported, called = executable
    assert isinstance(imported, ast.ImportFrom)
    assert imported.module == "hcu_adapter.bootstrap"
    assert [alias.name for alias in imported.names] == ["install"]
    assert isinstance(called, ast.Expr) and isinstance(called.value, ast.Call)
    assert isinstance(called.value.func, ast.Name)
    assert called.value.func.id == "install"


def test_gate_envs_documented():
    """对账契约：routing glue 裸读的门控 env（HCU_*/SGLANG_*/VLLM_*）
    必须以反引号全名出现在 docs/env-vars.md——多一漏一即红（R-12 层1，
    与 L0 的 REGISTRY→docs 方向互补，这里是 sitecustomize→docs 方向）。"""
    src = "\n".join(
        open(path, encoding="utf-8").read()
        for path in (SC_PATH, SGLANG_PATH)
    )
    names = set(_ENV_CALL.findall(src)) | set(_ENV_SUBSCRIPT.findall(src))
    assert names, "env extraction regex found nothing — pattern drifted?"
    doc = open(os.path.join(ROOT, "docs", "env-vars.md"),
               encoding="utf-8").read()
    undocumented = sorted(n for n in names if f"`{n}`" not in doc)
    assert not undocumented, (
        f"gate envs read in routing glue but not documented "
        f"(backticked full name) in docs/env-vars.md: {undocumented}")


# ------------------------------------------------------- 子进程基础设施

def _clean_env(extra=None):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HCU_", "VLLM_", "SGLANG_"))}
    parts = [ROOT] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                      if p and p != ROOT]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra:
        env.update(extra)
    return env


def _sglang_on_child_path():
    """子进程臂的 sglang 可达性：沿 _clean_env 组装的 PYTHONPATH 查 sglang
    包目录。不做真实 import——pytest 进程自身已被 sitecustomize 自动
    import 污染（见文件头），探针子进程又贵（全链 import 秒级）；路径
    存在性与门禁场景等价（L0/L1 门自 export PYTHONPATH）。"""
    for seg in _clean_env()["PYTHONPATH"].split(os.pathsep):
        if seg and os.path.isfile(os.path.join(seg, "sglang", "__init__.py")):
            return True
    return False


SGLANG_REACHABLE = _sglang_on_child_path()


def _child(script, extra_env=None, timeout=240):
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=_clean_env(extra_env), cwd=ROOT, timeout=timeout)


# ------------------------------------------------------------- L2 no-op

@pytest.mark.skipif(IS_HIP, reason="no-op contract is the non-HIP arm")
def test_noop_on_non_hip():
    """非 HIP 解释器：import sitecustomize 干净返回，零 env 写入、零日志。"""
    p = _child(
        "import os\n"
        "import sitecustomize\n"
        "print('VP', os.environ.get('VLLM_PLUGINS'))\n")
    assert p.returncode == 0, p.stderr[-500:]
    assert "VP None" in p.stdout, p.stdout
    assert "HCU adapter" not in p.stderr, (
        f"non-HIP arm must be silent, got: {p.stderr[:300]}")


def test_sglang_reachability_probe_contract():
    """R-13 #6 守卫钉：可达性探测沿 _clean_env 组装的 PYTHONPATH 找 sglang
    包；PYTHONPATH 抽掉 sglang 段后必须报不可达（HIP 子进程臂据此 SKIP
    而非 bare-shell FAIL）。纯路径逻辑，任何机器可跑。"""
    with mock.patch.dict(os.environ, {}, clear=True):
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [ROOT, "/nonexistent/sglang-root"])
        assert _sglang_on_child_path() is False

    # 带真 sglang 的 PYTHONPATH：正可达（仅当模块头已判可达，否则跳过断言）
    if SGLANG_REACHABLE:
        assert _sglang_on_child_path() is True
    # 收集期一次性计算与现算一致
    assert SGLANG_REACHABLE == _sglang_on_child_path()


# ------------------------------------------- L3/L4/L5（HIP 容器臂）

@pytest.mark.skipif(
    not IS_HIP, reason="install smoke needs the HIP wheel")
@pytest.mark.skipif(
    not SGLANG_REACHABLE,
    reason="sglang not on the child PYTHONPATH (bare shell): subproc arm "
           "cannot import sglang.srt, skip instead of fail (R-13 #6 guard)")
def test_install_smoke_and_dump_wiring():
    """干净子进程 import 全链后：platform 语义翻转（is_cuda()=True on HIP）、
    vllm._custom_ops.rms_norm 重绑到 vllm 宿主桥、VLLM_PLUGINS 被
    setdefault 成 "none"、HCU_ADAPTER_DUMP 布线落盘 registry.dump
    （ops/unbound/env 三键；import 期 ops 尚空是预期——dump 在链首
    log_summary 期写）。"""
    with tempfile.TemporaryDirectory() as td:
        dump = os.path.join(td, "dump.json").replace("\\", "/")
        p = _child(
            "import os, json\n"
            "import sitecustomize\n"
            "from sglang.srt.utils import is_cuda\n"
            "import vllm._custom_ops as vco\n"
            "d = json.load(open(os.environ['HCU_ADAPTER_DUMP']))\n"
            "print('RESULT', json.dumps({\n"
            "  'vp': os.environ.get('VLLM_PLUGINS'),\n"
            "  'is_cuda': bool(is_cuda()),\n"
            "  'rms_mod': getattr(vco.rms_norm, '__module__', None),\n"
            "  'dump_keys': sorted(d.keys()),\n"
            "  'mqa_src': d['env'].get('HCU_MQA_LOGITS_IMPL', {}).get('source'),\n"
            "}))\n",
            extra_env={"HCU_ADAPTER_DUMP": dump,
                       "HCU_MQA_LOGITS_IMPL": "triton"})
        assert p.returncode == 0, p.stderr[-800:]
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")),
                    None)
        assert line, f"no RESULT line; stdout={p.stdout[:400]}; stderr={p.stderr[:400]}"
        r = json.loads(line[len("RESULT "):])
        assert r["vp"] == "none", r
        assert r["is_cuda"] is True, r
        assert r["rms_mod"] == "hcu_adapter.bridges.vllm", r
        assert r["dump_keys"] == ["env", "ops", "unbound"], r
        assert r["mqa_src"] == "explicit", r  # L4: 显式 env 赢 setdefault


@pytest.mark.skipif(
    not IS_HIP, reason="VLLM_PLUGINS arm needs the HIP wheel")
@pytest.mark.skipif(
    not SGLANG_REACHABLE,
    reason="sglang not on the child PYTHONPATH (bare shell): subproc arm "
           "cannot import sglang.srt, skip instead of fail (R-13 #6 guard)")
def test_vllm_plugins_explicit_wins():
    """L4 双臂之二：VLLM_PLUGINS 预设时 setdefault 不覆盖（操作员 opt-in
    语义——"" 与 unset 有竞态陷阱，见 sitecustomize 门内注释）。"""
    p = _child(
        "import os\n"
        "import sitecustomize\n"
        "print('VP', os.environ.get('VLLM_PLUGINS'))\n",
        extra_env={"VLLM_PLUGINS": "myp"})
    assert p.returncode == 0, p.stderr[-500:]
    assert "VP myp" in p.stdout, p.stdout


if __name__ == "__main__":
    hip_only = (test_install_smoke_and_dump_wiring,
                test_vllm_plugins_explicit_wins)
    fails = 0
    for fn in (test_structure_gate_order_and_isolation,
               test_gate_envs_documented, test_noop_on_non_hip,
               test_install_smoke_and_dump_wiring,
               test_vllm_plugins_explicit_wins):
        if fn in hip_only and not IS_HIP:
            print(f"SKIP {fn.__name__} (non-HIP interpreter)")
            continue
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    print("RESULT:", "PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)
