# SPDX-License-Identifier: Apache-2.0
"""Read operator ABI declarations from the module that supplies the ABI."""

from collections.abc import Mapping
from functools import lru_cache
import logging


@lru_cache(maxsize=128)
def supports_operator_contract(provider, name, version=1):
    """A version describes a specific operator contract across package builds."""
    declarations = getattr(provider, 'HCU_OPERATOR_CONTRACTS', None)
    declared = declarations.get(name) if isinstance(declarations, Mapping) else None
    if type(declared) is int and declared == version:
        return True
    logging.getLogger('vllm.hcu_adapter').info(
        'HCU operator contract unavailable: provider=%s operator=%s '
        'expected_version=%s declared_version=%s',
        provider.__name__, name, version, declared,
    )
    return False
