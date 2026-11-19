# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang baa9befa381511cd4d56f83980951e5a45fffc6d.
# See THIRD_PARTY_NOTICES.md. Only the request/block grid mapping changes.

"""FP32 Mamba verify-state copy with request/block IDs flattened on grid x."""

import triton
import triton.language as tl


@triton.jit
def _fused_mamba_state_scatter_flat_kernel(
    src_ptr,
    dst_ptr,
    # Raw index arrays (before index_select)
    dst_indices_raw_ptr,  # [total_requests] - state_indices_tensor
    step_indices_raw_ptr,  # [total_requests] - last_correct_step_indices or mamba_steps_to_track
    src_indices_raw_ptr,  # [total_requests] - verify scratch row for each request
    elem_per_entry: tl.constexpr,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    dst_layer_stride,
    dst_req_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused gather-scatter kernel with built-in masking.

    This kernel fuses the index_select operations by:
    1. Iterating over all requests (pid_req from 0 to total_requests-1)
    2. Checking if step_indices_raw[pid_req] >= 0 (valid mask)
    3. If valid, performing the scatter:
       dst[l, dst_indices_raw[pid_req], :] = src[l, pid_req, step_indices_raw[pid_req], :]

    Grid: (total_requests * ceil(elem_per_entry / BLOCK_SIZE), num_layers)
    """
    num_blocks: tl.constexpr = tl.cdiv(elem_per_entry, BLOCK_SIZE)
    pid = tl.program_id(0)
    pid_req = pid // num_blocks
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_block = (pid % num_blocks).to(tl.int64)

    # Load step index to check validity (step >= 0 means valid)
    step_idx = tl.load(step_indices_raw_ptr + pid_req).to(tl.int64)

    # Early exit if this request is not valid (step < 0)
    if step_idx < 0:
        return

    # Load destination index
    dst_idx = tl.load(dst_indices_raw_ptr + pid_req).to(tl.int64)

    src_idx = tl.load(src_indices_raw_ptr + pid_req).to(tl.int64)

    # Bounds check to avoid illegal memory access
    if not (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx < src_step_size)
    ):
        return

    # Compute base offsets
    src_offset = (
        pid_layer * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
    )
    dst_offset = pid_layer * dst_layer_stride + dst_idx * dst_req_stride

    # Compute element range for this block
    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elem_per_entry

    # Load from source and store to destination
    data = tl.load(src_ptr + src_offset + offsets, mask=mask)
    tl.store(dst_ptr + dst_offset + offsets, data, mask=mask)
