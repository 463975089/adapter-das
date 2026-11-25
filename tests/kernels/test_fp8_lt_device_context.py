# SPDX-License-Identifier: Apache-2.0
"""CPU ownership probes; these do not establish GPU numerical correctness."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from hcu_adapter.kernels import fp8_scaled_mm_cpp as kernel


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("result", [0, 1, -1])
def test_foreign_current_device_uses_tensor_device_and_restores_caller(
        monkeypatch, capturing, result):
    caller = torch.device("cuda:0")
    owner = torch.device("cuda:1")
    state = {"device": caller, "calls": 0}

    @contextmanager
    def device_context(device):
        previous = state["device"]
        state["device"] = device
        try:
            yield
        finally:
            state["device"] = previous

    def tensor(shape, dtype, pointer):
        return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype,
            device=owner, is_cuda=True, is_contiguous=lambda: True,
            numel=lambda: shape[0] * shape[1], data_ptr=lambda: pointer)

    a = tensor((7, 1024), torch.float8_e4m3fn, 101)
    w = tensor((1024, 1024), a.dtype, 102)
    sa = tensor((7, 1), torch.float32, 103)
    sw = tensor((1024, 1), torch.float32, 104)
    output = tensor((7, 1024), torch.bfloat16, 105)

    def capture_state():
        # The tensor's stream is capturing while the caller's stream is not,
        # and vice versa. Tuning the wrong stream can synchronize a live graph.
        return capturing if state["device"] == owner else not capturing

    def execute(m, n, k, ap, wp, sap, swp, op, stream, allow_tune):
        assert state["device"] == owner, "vendor handle would use wrong device"
        assert (m, n, k, ap, wp, sap, swp, op, stream) == (
            7, 1024, 1024, 101, 102, 103, 104, 105, 987)
        assert allow_tune is not capturing
        state["calls"] += 1
        return result

    library = lambda: SimpleNamespace(hcu_fp8_lt_run=execute)
    library.cache_info = lambda: SimpleNamespace(currsize=1)
    monkeypatch.setattr(kernel, "_library", library)
    monkeypatch.setattr(kernel, "_device_supported", lambda device: True)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capture_state)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=987))
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: output)

    if result < 0:
        with pytest.raises(RuntimeError, match="execution failed"):
            kernel.try_scaled_mm(a, w, sa, sw)
    else:
        assert kernel.try_scaled_mm(a, w, sa, sw) is (output if result else None)
    assert state == {"device": caller, "calls": 1}
