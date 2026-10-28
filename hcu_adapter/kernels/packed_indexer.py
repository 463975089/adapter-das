"""Compatibility import; implementation lives in packed_indexer_triton."""

import sys
from . import packed_indexer_triton as _implementation

sys.modules[__name__] = _implementation
