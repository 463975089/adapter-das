# SPDX-License-Identifier: Apache-2.0
"""Native accumulation order across variable rows and changing Graph inputs."""
import pytest
import torch

from hcu_adapter.kernels.router_stagger_triton import project, supported


@pytest.mark.parametrize('m',list(range(5))+list(range(32,65,2)))
@torch.inference_mode()
def test_native_logit_bytes_for_every_admitted_row_count(m):
    if not torch.cuda.is_available():pytest.skip('HCU required')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx938':
        pytest.skip('Qualified gfx938 required')
    torch.manual_seed(231141+m)
    x=torch.randn(m,5120,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(384,5120,device='cuda',dtype=torch.bfloat16)
    assert supported(x,w)
    if not m:
        y=project(x,w);assert y.shape==(0,384) and y.dtype==torch.float32
        return
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):project(x,w)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=project(x,w)
    for amplitude in (.01,.25,8.):
        x.normal_(std=amplitude);w.normal_(std=.014)
        w[:192].copy_(w[0].expand(192,-1))
        before=x.clone();weight_before=w.clone()
        graph.replay()
        reference=torch.mm(x,w.T,out_dtype=torch.float32)
        torch.testing.assert_close(out.view(torch.uint8),reference.view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(before,x,rtol=0,atol=0)
        torch.testing.assert_close(weight_before,w,rtol=0,atol=0)


@torch.inference_mode()
def test_unsupported_cpu_shape_and_dtype():
    assert not supported(torch.zeros(7,5120,dtype=torch.bfloat16),
                         torch.zeros(384,5120,dtype=torch.bfloat16))
