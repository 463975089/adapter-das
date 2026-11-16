import pytest
import torch
from hcu_adapter.kernels.qsa_scatter_triton import scatter_rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype,shape", [(torch.bfloat16,(11,2,8)), (torch.int64,(11,3)), (torch.float8_e4m3fn,(11,2,8))])
def test_qsa_scatter_reserved_duplicates_and_replay(dtype, shape):
    loc = torch.tensor([0, 1, 0, 3, 0, 5], device="cuda")
    dest = torch.zeros(shape, dtype=dtype, device="cuda")
    values = torch.arange(6 * dest[0].numel(), device="cuda").reshape(6,*shape[1:]).to(dtype)
    scatter_rows(dest, loc, values)
    torch.testing.assert_close(dest.float()[[1,3,5]], values.float()[[1,3,5]], rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scatter_rows(dest, loc, values)
    loc.copy_(torch.tensor([0, 2, 0, 4, 0, 7], device="cuda"))
    values.copy_(torch.full_like(values, 3))
    graph.replay()
    torch.testing.assert_close(dest.float()[[2,4,7]], values.float()[[1,3,5]], rtol=0, atol=0)
