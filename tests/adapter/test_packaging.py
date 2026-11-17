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

# [L1 cold-curation 2026-11-17] 7 -> 3: keeps the stamped-pair interlock, the pure-shadow kernel wheel and the half-install sentinel; dropped faces stay in git history.
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


def _build_wheel(source_dir, tmp_path, tag, *, stamp=None, platform=None):
    wheel_dir = tmp_path / ("wheel-" + tag)
    wheel_dir.mkdir()
    env = os.environ.copy()
    env["HCU_ADAPTER_HOST"] = "none"
    env.pop("DAS_BUILD_STAMP", None)
    env.pop("DAS_PLATFORM_TAG", None)
    if stamp is not None:
        env["DAS_BUILD_STAMP"] = stamp
    if platform is not None:
        env["DAS_PLATFORM_TAG"] = platform
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


def _metadata(wheel):
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist()
                    if n.endswith(".dist-info/METADATA"))
        return Parser().parsestr(archive.read(name).decode())


def _interlock_pin(metadata):
    return next(Requirement(r) for r in metadata.get_all("Requires-Dist", [])
                if Requirement(r).name == "sgl-kernel")


def _pin_lib():
    # setup_hip.py and ci/l0_gate.py load this same renderer by path
    import importlib.util
    path = ADAPTER_PYPROJECT.parent / "_pin.py"
    spec = importlib.util.spec_from_file_location("hcu_adapter_pin_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_platform_tagged_pair_pins_exact_matching_build(tmp_path):
    # native kernel build: both wheels carry the dtk/torch ABI label
    # (no hcu segment) and the adapter pins that exact build.  The kernel
    # wheel itself comes from setup_hip.py (hipcc required); its version is
    # rendered by the same _pin.py the native setup loads, so exercising
    # that renderer here keeps this a hermetically checkable contract.
    source = _copy_repo(tmp_path)
    stamp = "20261111.g123abc"
    platform = "dtk2610.torch2130"
    adapter = _build_wheel(source / "packaging/hcu-adapter", tmp_path,
                           "platform-adapter", stamp=stamp, platform=platform)
    expected_kernel = f"{Version(_kernel_version()).public}+{platform}.{stamp}"
    expected_adapter = f"{Version(_adapter_version()).public}+{platform}.{stamp}"
    assert _pin_lib().kernel_version(stamp=stamp, platform=platform) == expected_kernel
    adapter_meta = _metadata(adapter)
    assert adapter_meta["Version"] == expected_adapter
    assert "hcu" not in expected_kernel
    pin = _interlock_pin(adapter_meta)
    assert str(pin.specifier) == f"=={expected_kernel}"
    assert pin.specifier.contains(expected_kernel)
    assert not pin.specifier.contains(f"{Version(_kernel_version()).public}+{platform}.other")


def test_platform_tag_without_stamp_is_still_an_exact_pin(tmp_path):
    source = _copy_repo(tmp_path)
    platform = "dtk2610.torch2130"
    adapter = _build_wheel(source / "packaging/hcu-adapter", tmp_path,
                           "platform-nostamp-adapter", platform=platform)
    expected_kernel = f"{Version(_kernel_version()).public}+{platform}"
    expected_adapter = f"{Version(_adapter_version()).public}+{platform}"
    assert _pin_lib().kernel_version(stamp="", platform=platform) == expected_kernel
    adapter_meta = _metadata(adapter)
    assert adapter_meta["Version"] == expected_adapter
    pin = _interlock_pin(adapter_meta)
    assert str(pin.specifier) == f"=={expected_kernel}"
    assert not pin.specifier.contains(f"{Version(_kernel_version()).public}+{platform}.other")


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
