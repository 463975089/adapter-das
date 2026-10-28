"""Compatibility alias for the common registry module."""

import sys
from hcu_adapter import registry as _implementation

sys.modules[__name__] = _implementation
