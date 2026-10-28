"""Compatibility import; implementation lives in vision_attention_emu."""

import sys
from . import vision_attention_emu as _implementation

sys.modules[__name__] = _implementation
