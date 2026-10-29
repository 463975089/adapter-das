"""Shared dispatch accounting for the LightOp operator families."""

_COUNTS = {}


def _count(name):
    _COUNTS[name] = _COUNTS.get(name, 0) + 1


def usage():
    return dict(_COUNTS)
