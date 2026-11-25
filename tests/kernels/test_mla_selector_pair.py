"""Integer exactness, independent formats/strides and changed graph metadata."""
import pytest
import torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='HCU required')

@pytest.fixture(scope='module',autouse=True)
def gfx938():
    if getattr(torch.cuda.get_device_properties(0),'gcnArchName','').split(':')[0]!='gfx938':
        pytest.skip('qualified gfx938 fused selector banks')

@pytest.mark.parametrize('rows',[1,7,17,42,48,129])
@pytest.mark.parametrize('widths',[(32,512),(65,1024),(128,2048)])
@pytest.mark.parametrize('csr',[(False,False),(True,True),(True,False),(False,True)])
@torch.inference_mode()
def test_native_eager_and_changing_graph(rows,widths,csr):
    from hcu_adapter.kernels.mla_selector_pair_triton import try_selector_banks
    from hcu_adapter.kernels.mla_sparse_decode_dtk import _selectors_eager
    from hcu_adapter.kernels.mla_selectors_triton import selectors
    values=[];banks=[]
    for width,is_csr in zip(widths,csr):
        ids=torch.empty(rows*width*2,device='cuda',dtype=torch.int32)[::2]
        lens=torch.empty(rows*2,device='cuda',dtype=torch.int32)[::2]
        ptr=torch.empty((rows+1)*2,device='cuda',dtype=torch.int32)[::2]
        values.append((ids,lens,ptr,width))
        banks.append((None,None,ids,ptr) if is_csr else (ids.reshape(rows,width),lens,None,None))
    def mutate(turn):
        for ids,lens,ptr,width in values:
            ids.random_(-3,517)
            lens.copy_(torch.arange(rows,device='cuda',dtype=torch.int32)*(turn+7)%(width+9)-2)
            ptr.copy_(torch.arange(rows+1,device='cuda',dtype=torch.int32)*width+(turn-2)*3)
    def call():return try_selector_banks(*banks,rows,509,511)
    def check(actual):
        for bank,got,slots in zip(banks,actual,(509,511)):
            for reference in (selectors(*bank,rows,slots),_selectors_eager(*bank,rows,slots)):
                for x,y in zip(got,reference):assert torch.equal(x,y)
    mutate(0);actual=call()
    if actual is None:
        assert rows==129 and widths==(128,2048);return
    check(actual)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):call()
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=call()
    for turn in (1,2,3):
        mutate(turn);graph.replay();torch.cuda.synchronize();check(out)

@torch.inference_mode()
def test_empty_and_large_shapes_keep_original_selector_path():
    from hcu_adapter.kernels.mla_selector_pair_triton import try_selector_banks
    x=torch.ones(2,1,device='cuda',dtype=torch.int32);n=torch.ones(2,device='cuda',dtype=torch.int32)
    assert try_selector_banks((x,n,None,None),(x[:,:0],n,None,None),2,128,128) is None
    x=torch.ones(2,4097,device='cuda',dtype=torch.int32)
    assert try_selector_banks((x,n,None,None),(x,n,None,None),2,8192,8192) is None
