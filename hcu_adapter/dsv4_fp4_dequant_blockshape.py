"""Compatibility entry for runtime FP4-to-FP8 block metadata adaptation.

The shared loader preserves aliases and never writes community fp8.py.
"""


def install():
    from .fp4_expert_load import install as install_runtime

    install_runtime()
