import sys
from pathlib import Path

from setuptools import setup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _pin  # noqa: E402

setup(
    version=_pin.adapter_version(),
    install_requires=[_pin.kernel_pin(), *_pin.base_dependencies()],
)
