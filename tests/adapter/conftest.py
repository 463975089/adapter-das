# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Host scoping for the vLLM-side adapter tests.

The vLLM-host suite (``test_vllm_*.py``, ``cases/vllm_*.py`` and the
vLLM-Worker ``test_model_mutation.py``) asserts process-level isolation
-- its ``isolated`` fixtures demand a CUDA-clean interpreter and the
bridges installed for host ``vllm``. Inside the sglang L1 kernels segment
one pytest process hosts sglang and has initialized CUDA on the device
tests, so on images that ship BOTH frameworks (2026-11-16 sglang:0.5.22
image carries vllm 0.33) these tests cannot run faithfully: they errored
en masse at setup (``assert not torch.cuda.is_initialized()``) instead of
being skipped as they were on vllm-less images. They still run -- and
must stay green -- in the vllm lane, where the host is pinned to vllm
and each file gets a fresh interpreter.
"""

import os

import pytest

# vLLM-Worker machinery without the test_vllm_ prefix.
_EXTRA_VLLM_FILES = {"test_model_mutation.py"}


def _is_vllm_host_file(path: str) -> bool:
    name = os.path.basename(path)
    if name in _EXTRA_VLLM_FILES:
        return True
    if name.startswith("test_vllm_"):
        return True
    parent = os.path.basename(os.path.dirname(path))
    return parent == "cases" and name.startswith("vllm_")


def pytest_collection_modifyitems(config, items):
    active = os.environ.get("_HCU_ADAPTER_ACTIVE_HOST")
    if active == "vllm":
        return
    reason = (
        "vllm-host suite: needs a CUDA-clean process with the vllm host "
        f"bridges (active host here: {active or 'none'})"
    )
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if _is_vllm_host_file(str(item.path)):
            item.add_marker(skip)
