# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Explicit two-HIP-device regression; run with task-owned GPUs visible."""
import importlib.util
from pathlib import Path
import sys

import torch


def main():
    if not torch.version.hip or torch.cuda.device_count() < 2:
        raise RuntimeError("This explicit regression requires two assigned HIP GPUs")
    path = Path(__file__).resolve().parents[1] / "hcu_adapter/kernels/_paged_mqa_mask_triton.py"
    spec = importlib.util.spec_from_file_location("_actual_mask_multidevice", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    mask = module.owned_logits_mask
    torch.cuda.set_device(0)
    with torch.cuda.device(1):
        original_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        scores = torch.ones((8, 1025), dtype=torch.float32, device="cuda:1")
        lengths = torch.full((8, 1), 1024, dtype=torch.int32, device="cuda:1")
        torch.cuda.synchronize(1)
        torch.cuda.set_stream(stream)
    try:
        # Warm up JIT and allocator before introducing a deliberately delayed
        # producer. Compilation latency must not hide a wrong-stream launch.
        warm = mask(scores, lengths)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        del warm
        for value in (7.0, 11.0, -3.0, 0.0):
            with torch.cuda.device(1):
                torch.cuda._sleep(80_000_000)
                scores.fill_(value)
            assert torch.cuda.current_device() == 0
            actual = mask(scores, lengths)
            assert torch.cuda.current_device() == 0
            assert torch.cuda.current_stream(1) == stream
            stream.synchronize()
            torch.cuda.synchronize(0)
            with torch.cuda.device(1):
                expected = torch.full_like(scores, value)
                expected[:, -1] = 0
                assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), (
                    "mask launch ignored the input device/current producer stream", value,
                )
            del actual
        print("MULTIDEVICE_CURRENT_STREAM_CHECKS=4", flush=True)
    finally:
        with torch.cuda.device(1):
            torch.cuda.set_stream(original_stream)
        torch.cuda.set_device(0)


if __name__ == "__main__":
    main()
