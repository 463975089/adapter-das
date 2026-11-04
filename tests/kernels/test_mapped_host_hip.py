import pytest
import torch
from hcu_adapter.mapped_host_dtk import _device_ptr_for_host_tensor, _wrap_mapped_host_as_cuda_tensor


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="HIP required")
@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16, torch.float8_e4m3fn])
def test_mapped_host_graph_reads_live_ids(dtype):
    host = torch.arange(128, dtype=torch.float32).reshape(16, 8).to(dtype).pin_memory()
    view, holder = _wrap_mapped_host_as_cuda_tensor(
        _device_ptr_for_host_tensor(host), tuple(host.shape), dtype, torch.device("cuda")
    )
    assert view.data_ptr() == holder.__cuda_array_interface__["data"][0]
    ids = torch.tensor([1, 4, 7], device="cuda")
    def gather():
        return torch.nn.functional.embedding(ids, view.view(torch.uint8)).view(dtype).to(torch.bfloat16)
    for _ in range(3):
        gather()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gather()
    for selected in ([9, 2, 15], [0, 5, 8], [13, 12, 11]):
        ids.copy_(torch.tensor(selected, device="cuda"))
        graph.replay()
        torch.testing.assert_close(output.cpu(), host[selected].to(torch.bfloat16), rtol=0, atol=0)
