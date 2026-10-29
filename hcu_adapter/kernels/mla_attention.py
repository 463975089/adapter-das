"""Compatibility import; implementation lives in mla_attention_emu."""

import sys
from . import mla_attention_emu as _implementation

sys.modules[__name__] = _implementation
