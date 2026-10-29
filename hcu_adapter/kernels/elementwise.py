"""Compatibility exports for the gather and norm operator families."""

from .gather import gather
from .gather_emu import gather as _torch_gather
from .norm import norm
from .norm_emu import norm as _torch_norm
