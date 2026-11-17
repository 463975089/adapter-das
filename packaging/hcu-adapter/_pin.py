"""Build-time version pin shared by setup.py and ci/l0_gate.py.

packaging/build.sh builds the native sgl-kernel via setup_hip.py and exports
DAS_PLATFORM_TAG=<dtk..>.torch.. — the compiled extension is bound to that
DTK/torch ABI, so both wheels of one build carry <platform>[.<stamp>] as the
local version and hcu-adapter depends on the exact tagged kernel build.
DAS_BUILD_STAMP (<time>.g<commit>) adds the per-build suffix; without it the
platform label alone is still an exact pin. Pure-python builds made directly
with `pip wheel` (tests/dev) have no platform tag and keep the old
hcu.<stamp> label; without a stamp the adapter falls back to its VERSION file
and the kernel's public version.
"""
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_VERSION_FILE = HERE.parent.parent / "sgl_kernel" / "version.py"


def build_stamp():
    return os.environ.get("DAS_BUILD_STAMP", "").strip().strip(".") or None


def platform_tag():
    return os.environ.get("DAS_PLATFORM_TAG", "").strip().strip(".") or None


def _local(stamp, platform):
    if platform:
        return f"{platform}.{stamp}" if stamp else platform
    return f"hcu.{stamp}" if stamp else ""


def kernel_public_version():
    src = KERNEL_VERSION_FILE.read_text(encoding="utf-8")
    m = re.search(r'^__version__ = "(.+?)"$', src, re.M)
    if not m:
        raise RuntimeError(f"cannot read __version__ from {KERNEL_VERSION_FILE}")
    return m.group(1).split("+", 1)[0]


def kernel_version(stamp=None, platform=None):
    if stamp is None:
        stamp = build_stamp()
    if platform is None:
        platform = platform_tag()
    local = _local(stamp, platform)
    version = kernel_public_version()
    return f"{version}+{local}" if local else version


def kernel_pin(stamp=None, platform=None):
    return f"sgl-kernel=={kernel_version(stamp=stamp, platform=platform)}"


def adapter_version(stamp=None, platform=None):
    if stamp is None:
        stamp = build_stamp()
    if platform is None:
        platform = platform_tag()
    version = (HERE / "VERSION").read_text(encoding="utf-8").strip()
    local = _local(stamp, platform)
    return f"{version.split('+', 1)[0]}+{local}" if local else version


def base_dependencies():
    """Validated DTK stack pins carried by the adapter dist.

    Mirrors the release line: a plain adapter install must pull the tested
    torch/triton builds instead of the newest nightly archive versions.
    """
    return ["packaging>=23", "torch==2.13.0", "triton==3.7.0"]
