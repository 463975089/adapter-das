"""Installed Core consumer parity across changed graph inputs and metadata."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='HCU required')


@pytest.fixture(scope='module',autouse=True)
def qualified_native():
    from hcu_adapter.kernels.mla_decode_rope_triton import qualify_inverse_rope
    if getattr(torch.cuda.get_device_properties(0),'gcnArchName','').split(':')[0] != 'gfx938':
        pytest.skip('qualified gfx938 native inverse RoPE')
    assert qualify_inverse_rope(torch.device('cuda',0))


@pytest.mark.parametrize('m',[1,7,17,42,48,64,129,256])
@pytest.mark.parametrize('heads',[64,128])
@pytest.mark.parametrize('position_dtype',[torch.int32,torch.int64])
@pytest.mark.parametrize('with_extra',[False,True])
@torch.inference_mode()
def test_installed_core_and_changing_graph(m,heads,position_dtype,with_extra):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as core
    from hcu_adapter.kernels.mla_decode_rope_triton import try_mask_copy_rope
    torch.manual_seed(231211)
    source=torch.empty(m,heads,512,device='cuda',dtype=torch.bfloat16)
    output=torch.empty_like(source)
    main=torch.empty(m,device='cuda',dtype=torch.int32)
    extra=torch.empty_like(main) if with_extra else None
    positions=torch.empty(m,device='cuda',dtype=position_dtype)
    angle=torch.randn(512,32,device='cuda')
    cs=torch.cat((angle.cos(),angle.sin()),-1)
    def mutate(turn):
        if turn==0:
            source.copy_(torch.arange(source.numel(),device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(source.shape))
        else:
            source.normal_(std=(.001,.25,8.)[(turn-1)%3])
        main.copy_(torch.arange(m,device='cuda',dtype=torch.int32)%(turn+3)-1)
        if extra is not None:extra.copy_(torch.arange(m,device='cuda',dtype=torch.int32)%(turn+2))
        positions.copy_((torch.arange(m,device='cuda')*(turn+1)+turn)%512)
    def reference():
        live=main>0 if extra is None else main+extra>0
        expected=torch.where(live[:,None,None],source,0.)
        core.rocm_inverse_rope_rows_(expected,positions,cs,64)
        return expected
    mutate(0)
    saved=source.clone()
    assert try_mask_copy_rope(source,output,main,extra,positions,cs)
    torch.testing.assert_close(output.view(torch.uint8),reference().view(torch.uint8),rtol=0,atol=0)
    torch.testing.assert_close(source.view(torch.uint8),saved.view(torch.uint8),rtol=0,atol=0)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):assert try_mask_copy_rope(source,output,main,extra,positions,cs)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        assert try_mask_copy_rope(source,output,main,extra,positions,cs)
    for turn in (1,2,3):
        mutate(turn);graph.replay()
        torch.testing.assert_close(output.view(torch.uint8),reference().view(torch.uint8),rtol=0,atol=0)


@torch.inference_mode()
def test_unsupported_preserves_output():
    from hcu_adapter.kernels.mla_decode_rope_triton import try_mask_copy_rope
    result=torch.zeros(2,16,512,device='cuda',dtype=torch.bfloat16)
    output=torch.full_like(result,3)
    lengths=torch.ones(2,device='cuda',dtype=torch.int32)
    pos=torch.zeros(2,device='cuda',dtype=torch.int64)
    cs=torch.ones(1,64,device='cuda')
    assert not try_mask_copy_rope(result,output,lengths,None,pos,cs)
    assert (output==3).all()
