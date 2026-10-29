"""Compatibility alias for the common hcu_config module."""

import sys
from hcu_adapter import hcu_config as _implementation

sys.modules[__name__] = _implementation
