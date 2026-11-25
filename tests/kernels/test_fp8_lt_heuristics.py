"""Installed native provider, graph ownership and metadata fallback contracts."""
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='HCU required')

@pytest.fixture(scope='module',autouse=True)
def gfx938():
    if getattr(torch.cuda.get_device_properties(0),'gcnArchName','').split(':')[0]!='gfx938':
        pytest.skip('qualified gfx938 vendor family')

@pytest.mark.parametrize('m',[1,7,17,35,42,48,64,65,129,256])
@pytest.mark.parametrize('shape',[(1024,1024),(2048,4096)])
@torch.inference_mode()
def test_native_provider_changed_graph_and_streams(m,shape):
    from hcu_adapter.kernels.fp8_scaled_mm_cpp import try_scaled_mm
    from hcu_adapter.kernels.scaled_mm_lightop import scaled_mm
    from lightop.gemm_ops import hipblaslt_w8a8_channelwise_gemm
    n,k=shape;torch.manual_seed(240000+m+n)
    a=torch.randn(m,k,device='cuda').to(torch.float8_e4m3fn)
    w=torch.randn(n,k,device='cuda').to(a.dtype)
    sa=torch.rand(m,1,device='cuda')*.1+.001
    sw=torch.rand(n,1,device='cuda')*.1+.001
    def call():return try_scaled_mm(a,w,sa,sw)
    def native():return scaled_mm(a,w.T,sa,sw,torch.bfloat16,None,gemm=hipblaslt_w8a8_channelwise_gemm)
    first=call();assert first is not None
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):call()
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):result=call()
    for repeat in range(3):
        a.copy_((torch.randn_like(a,dtype=torch.float32)*(repeat+1)).to(a.dtype))
        sa.uniform_(.001,.1);sw.uniform_(.001,.1)
        eager=call();expected=native();graph.replay();torch.cuda.synchronize()
        assert torch.equal(eager.view(torch.uint8),result.view(torch.uint8))
        af,bf=result.float(),expected.float()
        assert float((af-bf).norm()/bf.norm())<=.005
        assert float(torch.nn.functional.cosine_similarity(af.flatten(),bf.flatten(),dim=0))>=.99999
        for _ in range(5):assert torch.equal(call().view(torch.uint8),eager.view(torch.uint8))

@pytest.mark.parametrize('m,n,k,dtype',[(0,1024,1024,torch.float8_e4m3fn),
    (257,1024,1024,torch.float8_e4m3fn),(7,128,256,torch.float8_e4m3fn),
    (7,1024,1000,torch.float8_e4m3fn),(7,1024,1024,torch.int8)])
@torch.inference_mode()
def test_unsupported_is_untouched(m,n,k,dtype):
    from hcu_adapter.kernels.fp8_scaled_mm_cpp import try_scaled_mm
    a=torch.zeros(m,k,device='cuda',dtype=dtype);w=torch.ones(n,k,device='cuda',dtype=dtype)
    sa=torch.ones(m,1,device='cuda');sw=torch.ones(n,1,device='cuda')
    assert try_scaled_mm(a,w,sa,sw) is None
    assert (a.float()==0).all() and (w.float()==1).all()

@torch.inference_mode()
def test_cold_capture_returns_native_without_query_or_tuning():
    from hcu_adapter.kernels.fp8_scaled_mm_cpp import try_scaled_mm
    a=torch.ones(23,1152,device='cuda').to(torch.float8_e4m3fn)
    w=torch.ones(1088,1152,device='cuda').to(a.dtype)
    sa=torch.ones(23,1,device='cuda');sw=torch.ones(1088,1,device='cuda')
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):assert try_scaled_mm(a,w,sa,sw) is None
    graph.replay()


def test_native_fallback_handle_is_initialized_in_fresh_process():
    import subprocess
    import sys
    from pathlib import Path
    pytest.importorskip('vllm')
    script=Path(__file__).resolve().parents[1]/'tools/_probe_fp8_lt_cold_capture.py'
    result=subprocess.run([sys.executable,str(script)],stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,timeout=180)
    assert result.returncode==0,result.stdout
    assert '3 changed-input replays passed' in result.stdout
