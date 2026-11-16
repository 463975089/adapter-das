# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Bridge activation rebind guards: package-face dispatch, never a JIT compile.

Regression guard for ``hcu_adapter.bridges.sglang._bind_deepgemm_legacy_silu``.
Upstream's DeepGEMM/DeepEP runner computes ``_legacy_silu_and_mul = silu_and_mul``
only on CUDA, so HIP runs carry ``None`` and the bridge binds it. The binding
must go through the PACKAGE face (``sglang.kernels.ops.activation``) so the
fused-op dispatch resolves on HIP to AOT (shadow ``sgl_kernel``) instead of
importing the JIT module body and triggering a ninja build
(``HCU JIT: built sgl_kernel_jit_activation_... in 33s``, L1 kernels segment).

Three layers, because "the bridge bound something" is not the contract:

1. package face (always runs, no sglang needed): fake host modules prove the
   helper binds the package-face object, is idempotent, and never imports or
   touches the ``...activation.activation`` module body;
2. resolution (real sglang, still CPU-only): a mocked HIP platform makes the
   real package dispatch pick AOT; a mocked CUDA platform picks JIT (control:
   the mock really drives the seam, so a platform regression turns this red);
   one call of the bound symbol compiles nothing and leaves the JIT cache empty.
   Note the adapter answers CUDA semantics for sglang: the capability bridge in
   ``bridges/sglang.py`` scopes that CUDA->HIP relaxation out of fused-op
   dispatch, so HIP still filters the CUDA-only JIT candidate (this guard is
   what keeps that scoping honest);
3. tier (clean subprocess): ``HCU_SILU_MUL_IMPL=lightop`` / ``torch`` still
   resolve inside the shadow AOT op, so binding the package face keeps the
   tier switch reachable.

Run: HCU_ADAPTER_HOST=none python -m pytest -q tests/adapter/test_bridge_activation_rebind.py
"""

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

_PKG = "sglang.kernels.ops.activation"
_BODY = _PKG + ".activation"
_RUNNER = "sglang.srt.layers.moe.moe_runner.deep_gemm"
_ROOT = Path(__file__).resolve().parents[2]
_ABSENT = object()


class _TraceFinder:
    """Record (and then defer) imports of the forbidden module body."""

    def __init__(self, names):
        self.names = set(names)
        self.hits = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.names:
            self.hits.append(fullname)
        return None


def _module(name):
    module = types.ModuleType(name)
    # Package-ish so nested imports consult sys.modules instead of the finder.
    module.__path__ = []
    return module


def _install_fake_host(monkeypatch, *, package=_ABSENT, runner=_ABSENT):
    """Register a minimal ``sglang`` host tree; return the runner leaf module.

    ``sglang`` itself is faked: this path must also run where sglang is not
    installed (L0). The real-sglang case lives in the resolution test below.
    """
    names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.moe_runner",
        "sglang.kernels",
        "sglang.kernels.ops",
    )
    mods = {name: _module(name) for name in names}
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)

    mods["sglang"].srt = mods["sglang.srt"]
    mods["sglang.srt"].layers = mods["sglang.srt.layers"]
    mods["sglang.srt.layers"].moe = mods["sglang.srt.layers.moe"]
    mods["sglang.srt.layers.moe"].moe_runner = mods["sglang.srt.layers.moe.moe_runner"]
    mods["sglang"].kernels = mods["sglang.kernels"]
    mods["sglang.kernels"].ops = mods["sglang.kernels.ops"]

    if package is not _ABSENT:
        act = _module(_PKG)
        act.silu_and_mul = package
        mods[_PKG] = act
        mods["sglang.kernels.ops"].activation = act
        monkeypatch.setitem(sys.modules, _PKG, act)
    if runner is not _ABSENT:
        leaf = _module(_RUNNER)
        leaf._legacy_silu_and_mul = runner
        mods[_RUNNER] = leaf
        mods["sglang.srt.layers.moe.moe_runner"].deep_gemm = leaf
        monkeypatch.setitem(sys.modules, _RUNNER, leaf)
        return leaf
    # A booted adapter may have imported the real runner already; the fake
    # parent chain cannot shadow a fully-qualified sys.modules hit, so drop it.
    monkeypatch.delitem(sys.modules, _RUNNER, raising=False)
    return None


def _install_fake_runner_only(monkeypatch, runner_value):
    """Fake only the runner leaf/binaries; the real sglang kernels stay live."""
    sglang = importlib.import_module("sglang")
    names = (
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.moe_runner",
    )
    mods = {name: _module(name) for name in names}
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
    mods["sglang.srt"].layers = mods["sglang.srt.layers"]
    mods["sglang.srt.layers"].moe = mods["sglang.srt.layers.moe"]
    mods["sglang.srt.layers.moe"].moe_runner = mods[
        "sglang.srt.layers.moe.moe_runner"
    ]
    monkeypatch.setattr(sglang, "srt", mods["sglang.srt"], raising=False)

    leaf = _module(_RUNNER)
    leaf._legacy_silu_and_mul = runner_value
    mods["sglang.srt.layers.moe.moe_runner"].deep_gemm = leaf
    monkeypatch.setitem(sys.modules, _RUNNER, leaf)
    return leaf


def _bridge():
    return importlib.import_module("hcu_adapter.bridges.sglang")


# ---------------------------------------------------------------------------
# 1. package face -- runs everywhere
# ---------------------------------------------------------------------------

def test_helper_binds_package_face_and_never_imports_the_module_body(monkeypatch):
    sentinel = object()
    body_was_imported = _BODY in sys.modules
    runner = _install_fake_host(monkeypatch, package=sentinel, runner=None)
    body = _TraceFinder({_BODY})
    monkeypatch.setattr(sys, "meta_path", [body, *sys.meta_path])

    assert _bridge()._bind_deepgemm_legacy_silu() is True
    assert runner._legacy_silu_and_mul is sentinel
    assert body.hits == []
    if not body_was_imported:
        assert _BODY not in sys.modules

    # Idempotent: an existing binding is left alone, so a later import of the
    # package cannot silently re-point the runner mid-serve.
    replacement = object()
    sys.modules[_PKG].silu_and_mul = replacement
    assert _bridge()._bind_deepgemm_legacy_silu() is True
    assert runner._legacy_silu_and_mul is sentinel


def test_helper_keeps_a_host_bound_implementation(monkeypatch):
    existing = object()
    _install_fake_host(monkeypatch, runner=existing)
    package = _TraceFinder({_PKG})
    monkeypatch.setattr(sys, "meta_path", [package, *sys.meta_path])

    assert _bridge()._bind_deepgemm_legacy_silu() is True
    assert sys.modules[_RUNNER]._legacy_silu_and_mul is existing
    assert package.hits == []


def test_helper_replaces_the_jit_module_body_binding(monkeypatch):
    def jit_body(input, out=None):
        raise AssertionError("the JIT module body must never be called")

    # Upstream's import-time branch binds this body whenever sglang answers
    # CUDA semantics (the adapter platform seam); None is only the pre-seam
    # boot order. The bridge must replace exactly this binding.
    jit_body.__module__ = _BODY
    sentinel = object()
    runner = _install_fake_host(monkeypatch, package=sentinel, runner=jit_body)
    body = _TraceFinder({_BODY})
    monkeypatch.setattr(sys, "meta_path", [body, *sys.meta_path])

    assert _bridge()._bind_deepgemm_legacy_silu() is True
    assert runner._legacy_silu_and_mul is sentinel
    assert body.hits == []


def test_helper_is_a_noop_when_the_runner_is_absent(monkeypatch):
    _install_fake_host(monkeypatch)  # parents only: no deep_gemm leaf
    assert _bridge()._bind_deepgemm_legacy_silu() is False


def test_helper_source_imports_the_package_face_only():
    tree = ast.parse((_ROOT / "hcu_adapter/bridges/sglang.py").read_text("utf-8"))
    helper = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_bind_deepgemm_legacy_silu"
    )
    modules = []
    for node in ast.walk(helper):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    assert _PKG in modules
    assert [name for name in modules if name.startswith(_PKG + ".")] == []


# ---------------------------------------------------------------------------
# 2. resolution -- real sglang dispatch, mocked platform, CPU only
# ---------------------------------------------------------------------------

def test_hip_platform_resolves_silu_to_aot_and_calls_without_jit(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    act = pytest.importorskip("sglang.kernels.ops.activation")
    fused_op = importlib.import_module("sglang.kernels.fused_op")
    spec = importlib.import_module("sglang.kernels.spec")

    hip = spec.PlatformInfo(device_type="hip")
    cuda = spec.PlatformInfo(device_type="cuda")
    monkeypatch.setattr(fused_op, "_platform", lambda: hip)

    op = act.SiluAndMulOp()
    # Structural availability is not the contract: JIT *is* implemented and
    # first in priority; only the HIP capability filter keeps it out.
    assert spec.KernelBackend.JIT in op.available_backends()
    assert spec.KernelBackend.AOT in op.available_backends()
    assert op.auto_selected_backend() is spec.KernelBackend.AOT
    assert op.dispatch_forward().__name__ == "forward_aot"

    # Control: the same mock on CUDA resolves to JIT. If a platform-seam
    # regression ever reports CUDA semantics on HIP, the real call flips back
    # to the 33s ninja build and this contrast is what notices.
    monkeypatch.setattr(fused_op, "_platform", lambda: cuda)
    assert op.auto_selected_backend() is spec.KernelBackend.JIT
    assert op.dispatch_forward().__name__ == "forward_jit"

    monkeypatch.setattr(fused_op, "_platform", lambda: hip)

    loader = pytest.importorskip("sglang.kernels.jit.utils.compile.loader")
    compiles = []
    real_load_jit = loader.load_jit

    def counting_load_jit(*args, **kwargs):
        compiles.append((args, kwargs))
        return real_load_jit(*args, **kwargs)

    monkeypatch.setattr(loader, "load_jit", counting_load_jit)

    # Isolate the JIT cache: any build artifact must land under this root.
    cache_root = tmp_path / "jit-cache"
    monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", str(cache_root))
    body_was_imported = _BODY in sys.modules

    runner = _install_fake_runner_only(monkeypatch, None)
    assert _bridge()._bind_deepgemm_legacy_silu() is True
    bound = runner._legacy_silu_and_mul
    assert bound is act.silu_and_mul

    # The shared op instance may have cached a dispatch from an earlier test;
    # reset it so this call resolves under the HIP mock. Also pin auto mode
    # (no SGLANG_FORCE_FUSED_OP_BACKEND from the surrounding gate).
    monkeypatch.setattr(act._SILU_AND_MUL, "_forward_method", None)
    monkeypatch.setattr(fused_op, "_forced_backend", None)

    inp = torch.randn(4, 128, dtype=torch.bfloat16)
    out = bound(inp, None)
    expected = torch.nn.functional.silu(inp[..., :64].float()) * inp[..., 64:].float()
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)

    assert compiles == []
    if not body_was_imported:
        assert _BODY not in sys.modules
    assert not cache_root.exists() or not any(cache_root.iterdir())


# ---------------------------------------------------------------------------
# 3. tier -- HCU_SILU_MUL_IMPL still reaches the shadow AOT op
# ---------------------------------------------------------------------------

_TIER_SCRIPT = r"""
import sys
import types

tier = sys.argv[1]
if tier == "lightop":
    fake = types.ModuleType("lightop")
    fake.fuse_silu_and_mul = lambda input, out: "lightop-sentinel"
    sys.modules["lightop"] = fake

from sgl_kernel import elementwise

assert elementwise._IMPL_CHOICE is None, elementwise._IMPL_CHOICE
assert elementwise._resolve_impl() == tier, elementwise._IMPL_CHOICE
if tier == "lightop":
    import lightop
    assert elementwise._LIGHTOP_IMPL is lightop.fuse_silu_and_mul
else:
    assert elementwise._LIGHTOP_IMPL is None
"""


@pytest.mark.parametrize("tier", ["lightop", "torch"])
def test_shadow_aot_op_still_honours_the_tier_switch(tier):
    env = dict(
        os.environ,
        HCU_ADAPTER_HOST="none",
        HCU_SILU_MUL_IMPL=tier,
        PYTHONPATH=str(_ROOT),
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("HCU_CONFIG", None)
    env.pop("HCU_CONFIG_FILE", None)
    result = subprocess.run(
        [sys.executable, "-c", _TIER_SCRIPT, tier],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
