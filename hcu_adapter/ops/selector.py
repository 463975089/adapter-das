import importlib

from . import config, kill_switch


def guard(op):
    op = op.lower()
    disabled = kill_switch.disabled_ops()
    if any(op == name or op.startswith(name + ".") for name in disabled):
        raise RuntimeError("HCU_DISABLE_OPS disables " + op)


def select(op, default, *, requested=None):
    guard(op)
    spec = config.BACKEND_CATALOG[op]
    backend = requested if requested is not None else config.load()["ops"].get(op, default)
    if backend not in spec["backends"]:
        raise ValueError(f"Unsupported backend {backend!r} for {op}")
    if spec["backends"][backend]["status"] in ("planned", "broken"):
        raise ValueError(f"Unavailable backend {backend!r} for {op}")
    return backend


def provider(op, default, *, requested=None):
    backend = select(op, default, requested=requested)
    path = config.BACKEND_CATALOG[op]["backends"][backend].get("implementation")
    if not path:
        raise ValueError("No standalone provider for " + op)
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)
