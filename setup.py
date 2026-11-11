import os
import re
from pathlib import Path

from setuptools import setup

HERE = Path(__file__).resolve().parent


def _version():
    src = (HERE / "sgl_kernel" / "version.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__ = "(.+?)"$', src, re.M)
    if not m:
        raise RuntimeError("cannot read __version__ from sgl_kernel/version.py")
    version = m.group(1)
    stamp = os.environ.get("DAS_BUILD_STAMP", "").strip().strip(".")
    return f"{version.split('+', 1)[0]}+hcu.{stamp}" if stamp else version


setup(version=_version())
