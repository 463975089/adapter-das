# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public-contract regression for the vLLM extension API (API1).

PR #142 review P1: the entry-point group name is a protocol published in
Plugin distribution metadata (adapter_extension v0.33.0 ships
``[project.entry-points."hcu_adapter.extensions"]``), NOT an import path --
it must not follow module moves. And out-of-tree Plugin code imports
``ExtensionSpec``/``ExtensionCommitError`` from ``hcu_adapter.extensions``
directly, so that module must stay importable at its published path.
"""

import importlib
import importlib.metadata
from importlib import metadata

import pytest

PUBLISHED_GROUP = "hcu_adapter.extensions"
PUBLISHED_MODULE = "hcu_adapter.extensions"


def test_entry_point_group_is_the_published_protocol_name():
    from hcu_adapter.infra.extensions import EXTENSION_ENTRY_POINT_GROUP

    assert EXTENSION_ENTRY_POINT_GROUP == PUBLISHED_GROUP


def test_published_module_imports_and_reexports_api1():
    pub = importlib.import_module(PUBLISHED_MODULE)
    impl = importlib.import_module("hcu_adapter.infra.extensions")
    for name in pub.__all__:
        assert getattr(pub, name) is getattr(impl, name), name
    # the exact names Plugin <= v0.33.0 imports directly
    assert issubclass(pub.ExtensionCommitError, RuntimeError)
    assert issubclass(pub.RequiredExtensionError, RuntimeError)
    assert pub.EXTENSION_API_VERSION == 1


def test_plugin_metadata_discovery_roundtrip():
    """Cross-repo leg: with a real Plugin distribution installed, every
    published entry point must be discoverable under the group AND load.
    Skipped where no plugin dist exists (plain CI lanes)."""
    eps = [
        ep
        for ep in metadata.entry_points()
        if getattr(ep, "group", None) == PUBLISHED_GROUP
    ]
    if not eps:
        pytest.skip("no plugin distribution installed in this environment")
    for ep in eps:
        ep.load()  # raises on a broken public face -- no try/except here
    assert len(eps) >= 1
