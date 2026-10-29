"""Explicit top-k provider for the pinned deep_select.interface seam."""

import sys
import types

from . import hcu_env
from hcu_adapter import registry


def install():
    from hcu_adapter.kernels.deepselect_emu import topk
    if hcu_env.lightop_ops_enabled():
        from hcu_adapter.kernels.deepselect_lightop import topk

    if "deep_select" in sys.modules:
        if getattr(sys.modules.get("deep_select.interface"), "topk", None) is topk:
            return
        raise RuntimeError("An existing deep_select provider is already imported")
    package = types.ModuleType("deep_select")
    package.__path__ = []
    interface = types.ModuleType("deep_select.interface")
    interface.topk = package.topk = topk
    interface.get_stride_requirement = package.get_stride_requirement = lambda: (1, 32)
    package.interface = interface
    sys.modules["deep_select"] = package
    sys.modules["deep_select.interface"] = interface
    registry.register("deep_select.topk", topk.__module__ + '.topk',
                      "deep_select.interface.topk", "seam")
    print("HCU adapter runtime: deep_select.topk -> " + topk.__module__ + " (candidate)")
