"""Torch/emu implementations of the pinned state_and_constraints contracts."""

import torch

def select_state_slice(state_tensor, state_cache, last_hit_position):
    infer = state_tensor.shape[1]
    pos = torch.arange(infer, device=state_tensor.device) + last_hit_position - infer
    state_tensor.index_copy_(1, pos.remainder(infer).long(),
                             state_cache.index_select(1, pos.remainder(state_cache.shape[1]).long()))


def load_state_to_cache(state_tensor, state_cache, last_hit_position):
    infer = state_tensor.shape[1]
    pos = torch.arange(infer, device=state_tensor.device) + last_hit_position - infer
    state_cache.index_copy_(1, pos.remainder(state_cache.shape[1]).long(),
                            state_tensor.index_select(1, pos.remainder(infer).long()))


def detect_outlier(data, identifier, max_abs):
    # DTK Triton's printf/assert runtime buffers are unavailable on this build:
    # the negative diagnostic probe VMFaulted. Use PyTorch's supported async
    # device assertion; value/offset diagnostics remain an explicit R-15 gap.
    torch._assert_async(((data.float() > -max_abs) & (data.float() < max_abs)).all(),
                        f"outlier detected: id={identifier}, permitted abs(value) < {max_abs}")
