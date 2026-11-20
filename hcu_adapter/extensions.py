# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Stable public import path for the vLLM extension API (API1).

The implementation lives in ``hcu_adapter/infra/extensions.py`` under layout
governance; this module is the PUBLISHED import path and must stay stable for
out-of-tree consumers. Plugin adapter_extension (<= v0.33.0) imports
``ExtensionSpec`` / ``ExtensionCommitError`` / ``ExtensionReport`` from
``hcu_adapter.extensions`` directly, and its distribution metadata publishes
entry points under the ``hcu_adapter.extensions`` group -- a protocol name in
installed package metadata, not an import path, so it does not follow file
moves (PR #142 review P1). Removing this shim or renaming the group breaks
every deployed Plugin at once; changes to the public face need a versioned
API bump plus coordinated releases, not a directory migration.
"""

from hcu_adapter.infra.extensions import (
    EXTENSION_API_VERSION,
    EXTENSION_ENTRY_POINT_GROUP,
    ExtensionContext,
    ExtensionSpec,
    ExtensionStatus,
    ExtensionReport,
    RequiredExtensionError,
    ExtensionCommitError,
    is_fatal_extension_error,
    activate_extensions,
)

__all__ = [
    "EXTENSION_API_VERSION",
    "EXTENSION_ENTRY_POINT_GROUP",
    "ExtensionContext",
    "ExtensionSpec",
    "ExtensionStatus",
    "ExtensionReport",
    "RequiredExtensionError",
    "ExtensionCommitError",
    "is_fatal_extension_error",
    "activate_extensions",
]
