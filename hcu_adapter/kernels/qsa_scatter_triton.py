import torch
import triton
import triton.language as tl

@triton.jit
def _qsa_scatter_rows_kernel(
    buf_ptr,
    loc_ptr,
    val_ptr,
    n_rows,
    ROW_ELEMS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    rmask = rows < n_rows
    dst = tl.load(loc_ptr + rows, mask=rmask, other=0).to(tl.int64)
    for c in tl.static_range(ROW_ELEMS):
        v = tl.load(val_ptr + rows * ROW_ELEMS + c, mask=rmask, other=0)
        tl.store(buf_ptr + dst * ROW_ELEMS + c, v, mask=rmask)


def scatter_rows(buffer, loc, values):
    loc = loc.long().reshape(-1)
    if loc.numel() != values.shape[0]:
        raise ValueError("QSA scatter row count mismatch")
    if not buffer.is_contiguous():
        raise ValueError("QSA scatter destination must be contiguous")
    if not loc.numel():
        return
    loc = loc.clamp(0, max(buffer.shape[0] - 1, 0))
    values = values.to(buffer.dtype).contiguous()
    if buffer.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        buffer = buffer.view(torch.uint8)
        values = values.view(torch.uint8)
    _qsa_scatter_rows_kernel[(triton.cdiv(loc.numel(), 128),)](
        buffer.view(-1), loc, values.view(-1), loc.numel(),
        ROW_ELEMS=values.numel() // values.shape[0], BLOCK_ROWS=128,
    )
