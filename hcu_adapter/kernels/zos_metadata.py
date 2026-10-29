"""Compatibility import; implementation lives in zos_metadata_emu."""

import sys
from . import zos_metadata_emu as _implementation

sys.modules[__name__] = _implementation
