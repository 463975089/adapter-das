"""Explicitly bridge the installed HCU DeepEP without replacing its package."""

from . import hcu_env
from hcu_adapter import registry


def _missing(name):
    class Unavailable:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("HCU provider missing for deep_ep." + name)

    Unavailable.__name__ = name
    return Unavailable


def install(allow_cuda_graph=False):
    import deep_ep
    from hcu_adapter.kernels import deepep_compat

    ep_type = deepep_compat.GraphEPBuffer if allow_cuda_graph else deepep_compat.EPBuffer
    engram_type = deepep_compat.GraphEngramBuffer if allow_cuda_graph else deepep_compat.EngramBuffer
    disabled = hcu_env.resolve("HCU_DISABLE_OPS")
    if any(op in disabled for op in ("deep_ep", "deep_ep.EPBuffer", "deep_ep.EngramBuffer")):
        raise RuntimeError("HCU_DISABLE_OPS disables the explicit DeepEP candidate")
    if hasattr(deep_ep, "EPBuffer"):
        if deep_ep.EPBuffer is ep_type and deep_ep.EngramBuffer is engram_type:
            return
        raise RuntimeError("An existing deep_ep.EPBuffer provider is already installed")
    deepep_compat.EPBuffer.vendor_type = deep_ep.Buffer
    deep_ep.EPBuffer = ep_type
    deep_ep.EPHandle = deepep_compat.EPHandle
    deep_ep.EngramBuffer = engram_type
    deep_ep.EventOverlap = deepep_compat.EventOverlap
    deep_ep.EventHandle = deep_ep.deep_ep_cpp.EventHandle
    from hcu_adapter.kernels import bucket

    for name in ("BucketBuffer", "BucketSession", "get_num_allocation_alignment", "get_num_rdma_alignment"):
        provider = getattr(bucket, name)
        if hasattr(deep_ep, name):
            raise RuntimeError("An existing deep_ep." + name + " provider is already installed")
        setattr(deep_ep, name, provider)
        registry.register("deep_ep." + name, "hcu_adapter.kernels.bucket." + name, "deep_ep." + name, "rebind")
    for name in ("get_physical_domain_size", "get_logical_domain_size"):
        if hasattr(deep_ep, name):
            raise RuntimeError("An existing deep_ep." + name + " provider is already installed")
        setattr(deep_ep, name, getattr(deepep_compat, name))
        registry.register("deep_ep." + name, "hcu_adapter.kernels.deepep_compat." + name,
                          "deep_ep." + name, "rebind")
    registry.register("deep_ep.EPBuffer", "hcu_adapter.kernels.deepep_compat." + ep_type.__name__,
                      "deep_ep.EPBuffer", "rebind")
    registry.register("deep_ep.EngramBuffer", "hcu_adapter.kernels.deepep_compat." + engram_type.__name__,
                      "deep_ep.EngramBuffer", "rebind")
    mode = "graph" if allow_cuda_graph else "eager"
    print(f"HCU adapter runtime: deep_ep.EPBuffer -> DTK v1 + expanded {mode} candidate", flush=True)
