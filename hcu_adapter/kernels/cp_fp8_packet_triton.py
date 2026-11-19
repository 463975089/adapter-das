# SPDX-License-Identifier: Apache-2.0
"""Pack unit-scale MiniMax K, V and index K for one CP collective."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(k, v, index, out, rows, ks, vs, ins, BLOCK_ROWS: tl.constexpr):
    row = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS))[:, None]
    cols = tl.arange(0, 512)[None, :]
    dst = out + row * 1152
    k_value = tl.load(k + row * ks + cols, row < rows, other=0)
    v_value = tl.load(v + row * vs + cols, row < rows, other=0)
    fp8 = out.dtype.element_ty
    k_value = tl.clamp(k_value.to(tl.float32), -448.0, 448.0,
                       propagate_nan=tl.PropagateNan.ALL)
    v_value = tl.clamp(v_value.to(tl.float32), -448.0, 448.0,
                       propagate_nan=tl.PropagateNan.ALL)
    tl.store(dst + cols, k_value.to(fp8, fp_downcast_rounding="rtne"), row < rows)
    tl.store(dst + 512 + cols, v_value.to(fp8, fp_downcast_rounding="rtne"), row < rows)
    index_cols = tl.arange(0, 128)[None, :]
    i_value = tl.load(index + row * ins + index_cols, row < rows, other=0)
    tl.store(dst + 1024 + index_cols, i_value.to(fp8, fp_downcast_rounding="rtne"), row < rows)


def pack_unit_scale_fp8_kv(k, v, index, k_scale, v_scale):
    if k_scale not in (None, 1.0) or v_scale not in (None, 1.0):
        raise ValueError("CP FP8 packet requires unit K and V scales")
    rows = k.shape[0]
    packet = torch.empty((rows, 1152), dtype=torch.float8_e4m3fn, device=k.device)
    _pack[(triton.cdiv(rows, 4),)](
        k, v, index, packet, rows, k.stride(0), v.stride(0), index.stride(0),
        BLOCK_ROWS=4, num_warps=4,
    )
    return packet.view(torch.uint8)
