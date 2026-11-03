"""Compatibility entry for runtime FP4 expert loading and explicit DeepEP BF16.

The shared loader preserves aliases and never writes community fp8.py.
"""


def install():
    from .fp4_expert_load import install as install_runtime

    install_runtime()
