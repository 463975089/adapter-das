# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Dual-dist packaging contracts (governance §7.4).

Two wheels, independent version lines, one interlock:
  sgl-kernel  = pure shadow kernel (sgl_kernel/* only; no sitecustomize,
                no entry-points); version single source:
                sgl_kernel/version.py (kernel-line pairing)
  hcu-adapter = hcu_adapter/* + sitecustomize.py + vllm entry-points;
                version single source: packaging/hcu-adapter/VERSION
                (sglang-line pairing); interlocks the kernel dist via
                Requires-Dist: sgl-kernel==<sgl_kernel version>
"""

import os
from email.parser import Parser
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10, supported by the package.
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PYPROJECT = REPO_ROOT / "packaging" / "hcu-adapter" / "pyproject.toml"


def _kernel_version():
    src = (REPO_ROOT / "sgl_kernel" / "version.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__ = "(.+?)"$', src, re.M)
    assert m, "sgl_kernel/version.py must carry __version__ (single source)"
    return m.group(1)


def _adapter_version():
    return (ADAPTER_PYPROJECT.parent / "VERSION").read_text(
        encoding="utf-8").strip()


def _build_wheel(source_dir, tmp_path, tag, *, stamp=None):
    wheel_dir = tmp_path / ("wheel-" + tag)
    wheel_dir.mkdir()
    env = os.environ.copy()
    env["HCU_ADAPTER_HOST"] = "none"
    env.pop("DAS_BUILD_STAMP", None)
    if stamp is not None:
        env["DAS_BUILD_STAMP"] = stamp
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(source_dir),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    return next(wheel_dir.glob("*.whl"))


def _copy_repo(tmp_path):
    source = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", "__pycache__", "build", "dist",
            "*.egg-info",
        ),
    )
    return source


@pytest.fixture(scope="module")
def kernel_wheel(tmp_path_factory):
    return _build_wheel(_copy_repo(tmp_path_factory.mktemp("src-kernel")),
                        tmp_path_factory.getbasetemp(), "kernel")


@pytest.fixture(scope="module")
def adapter_wheel(tmp_path_factory):
    # the adapter dist builds from packaging/hcu-adapter with sources left
    # at the repo root (src-layout mapping, nothing copied)
    source = _copy_repo(tmp_path_factory.mktemp("src-adapter"))
    return _build_wheel(source / "packaging" / "hcu-adapter",
                        tmp_path_factory.getbasetemp(), "adapter")


def _members(wheel):
    archive = zipfile.ZipFile(wheel)
    return set(archive.namelist()), archive


def test_kernel_wheel_is_pure_shadow_kernel(kernel_wheel):
    members, archive = _members(kernel_wheel)
    version = _kernel_version()

    expected = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "sgl_kernel").rglob("*.py")
    }
    assert not (expected - members), f"kernel wheel missing: {sorted(expected - members)}"
    # purity: no adapter payload rides the kernel dist
    assert "sitecustomize.py" not in members
    assert not any(m.startswith("hcu_adapter/") for m in members), \
        "kernel wheel must not ship hcu_adapter/"
    assert f"sgl_kernel-{version}.dist-info/METADATA" in members
    metadata = archive.read(f"sgl_kernel-{version}.dist-info/METADATA").decode()
    assert "Requires-Dist: torch" in metadata  # torch/triton/packaging stay
    ep = f"sgl_kernel-{version}.dist-info/entry_points.txt"
    assert ep not in members, "kernel dist must not carry vllm entry-points"
    archive.close()


def test_adapter_wheel_carries_bootstrap_bridges_and_entry_points(adapter_wheel):
    members, archive = _members(adapter_wheel)
    version = _adapter_version()

    expected = {"sitecustomize.py"}
    expected.update(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "hcu_adapter").rglob("*.py")
    )
    assert not (expected - members), f"adapter wheel missing: {sorted(expected - members)}"
    # purity: the kernel package does not ride the adapter dist
    assert not any(m.startswith("sgl_kernel/") for m in members), \
        "adapter wheel must not ship sgl_kernel/ (interlock pins the dist)"

    dist_info = f"hcu_adapter-{version}.dist-info"
    assert f"{dist_info}/METADATA" in members
    metadata = archive.read(f"{dist_info}/METADATA").decode()
    assert f"Requires-Dist: sgl-kernel=={Version(_kernel_version()).public}" in metadata, \
        "unstamped adapter must pin the kernel's public version"
    assert "sgl-kernel~=" not in metadata

    entry_points = archive.read(f"{dist_info}/entry_points.txt").decode()
    assert "[vllm.platform_plugins]" in entry_points
    assert "[vllm.general_plugins]" in entry_points
    assert "hcu_adapter = hcu_adapter.bridges.vllm:hcu_platform_plugin" in entry_points
    assert "hcu_adapter = hcu_adapter.bridges.vllm:hcu_runtime_plugin" in entry_points
    archive.close()


def test_vllm_entry_points_declared_only_in_adapter_dist():
    adapter_project = tomllib.loads(
        ADAPTER_PYPROJECT.read_text(encoding="utf-8")
    )["project"]
    entry_points = adapter_project["entry-points"]
    assert entry_points["vllm.platform_plugins"] == {
        "hcu_adapter": "hcu_adapter.bridges.vllm:hcu_platform_plugin"
    }
    assert entry_points["vllm.general_plugins"] == {
        "hcu_adapter": "hcu_adapter.bridges.vllm:hcu_runtime_plugin"
    }
    # entry-points migrated (not duplicated): the kernel dist declares none
    kernel_project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert "entry-points" not in kernel_project


def test_interlock_pin_matches_single_version_source(adapter_wheel):
    project = tomllib.loads(ADAPTER_PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert "dependencies" in project["dynamic"] and "dependencies" not in project
    with zipfile.ZipFile(adapter_wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(name).decode())
    base = {str(r) for line in metadata.get_all("Requires-Dist", [])
            if (r := Requirement(line)).marker is None}
    assert base == {f"sgl-kernel=={Version(_kernel_version()).public}",
                    "packaging>=23", "torch==2.13.0", "triton==3.7.0"}


def test_stamped_wheel_pair_pins_exact_matching_build(tmp_path):
    source = _copy_repo(tmp_path)
    stamp = "20261111.g123abc"
    kernel = _build_wheel(source, tmp_path, "stamped-kernel", stamp=stamp)
    adapter = _build_wheel(source / "packaging/hcu-adapter", tmp_path,
                           "stamped-adapter", stamp=stamp)
    messages = []
    for wheel in (kernel, adapter):
        with zipfile.ZipFile(wheel) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            messages.append(Parser().parsestr(archive.read(name).decode()))
    expected_kernel = f"{Version(_kernel_version()).public}+hcu.{stamp}"
    expected_adapter = f"{Version(_adapter_version()).public}+hcu.{stamp}"
    assert messages[0]["Version"] == expected_kernel
    assert messages[1]["Version"] == expected_adapter
    pin = next(Requirement(r) for r in messages[1].get_all("Requires-Dist", [])
               if Requirement(r).name == "sgl-kernel")
    assert str(pin.specifier) == f"=={expected_kernel}"
    assert pin.specifier.contains(expected_kernel)
    assert not pin.specifier.contains(f"{Version(_kernel_version()).public}+hcu.other")


def test_adapter_version_file_is_the_adapter_single_source(adapter_wheel):
    # packaging/hcu-adapter/VERSION is the adapter dist's own single source
    # (independent of sgl-kernel's line); the built wheel must carry it
    # verbatim in its metadata.
    members, archive = _members(adapter_wheel)
    dist_info = f"hcu_adapter-{_adapter_version()}.dist-info"
    metadata = archive.read(f"{dist_info}/METADATA").decode()
    archive.close()
    assert f"Version: {_adapter_version()}" in metadata


_SENTINEL_BLOCK = """\
import importlib.util as ilu
_real = ilu.find_spec
def _blocked(name, *a, **k):
    if name == "hcu_adapter" or name.startswith("hcu_adapter."):
        return None
    return _real(name, *a, **k)
ilu.find_spec = _blocked
try:
    import sgl_kernel
except RuntimeError as e:
    assert "hcu-adapter" in str(e), e
    assert "pip install hcu-adapter" in str(e), e
    print("SENTINEL-RAISED-OK")
else:
    raise SystemExit("sgl_kernel imported without hcu_adapter: sentinel missing")
"""


def test_sgl_kernel_half_install_raises_sentinel():
    # governance §7.4 sentinel: a kernel-dist-only environment must fail
    # loudly at import instead of silently running a half-adapted kernel.
    result = subprocess.run(
        [sys.executable, "-c", _SENTINEL_BLOCK],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "HCU_ADAPTER_HOST": "none"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SENTINEL-RAISED-OK" in result.stdout
