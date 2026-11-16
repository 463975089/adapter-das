"""Compatibility contract used by the opt-in vLLM V4.1 PCP extension."""
from hcu_adapter import registry


def register_compatibility():
    """Record scoped model dispatch without loading device code in the controller."""
    registry.register(
        op="vllm_dsv41_pcp_compat", impl="scoped", target=__name__,
        mechanism=registry.MECH_SEAM, source=__name__,
    )
