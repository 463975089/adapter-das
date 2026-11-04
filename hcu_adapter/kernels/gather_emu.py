# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Torch implementation of the existing gather contract."""

def gather(x, index, out):
    import torch

    if not x.shape[0]:
        out.zero_()
        return
    selected = torch.index_select(x, 0, index.clamp(0, x.shape[0] - 1))
    valid = (index >= 0) & (index < x.shape[0])
    out.copy_(torch.where(valid[:, None], selected, 0))
