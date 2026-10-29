"""Compatibility alias for the common kill_switch module."""

import sys
from hcu_adapter import kill_switch as _implementation

sys.modules[__name__] = _implementation
