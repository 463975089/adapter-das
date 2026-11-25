# SPDX-License-Identifier: Apache-2.0
"""Exact native storage/quantization boundaries and changing Graph inputs."""
import pytest
import torch

from hcu_adapter.kernels.qkv_norm_token_fp8_triton import norm_quantize


@pytest.mark.parametrize('m,q,k',[(0,1280,512),(1,1280,512),(7,1280,512),
    (42,1280,512),(48,1280,512),(65,1280,512),(257,1280,512),
    (3,128,64),(9,640,256),(17,2048,512)])
@pytest.mark.parametrize('lightop',[False,True])
@torch.inference_mode()
def test_native_norm_quant_bytes_and_changing_graph(m,q,k,lightop):
    if not torch.cuda.is_available():pytest.skip('HCU required')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx938':
        pytest.skip('Qualified gfx938 required')
    native=pytest.importorskip('vllm.models.common.ops.fused_qk_rmsnorm')
    from vllm import _custom_ops as ops
    from vllm.config import CompilationConfig,VllmConfig,set_current_vllm_config
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    torch.manual_seed(231138)
    backing=torch.empty((m,q+k+13),device='cuda',dtype=torch.bfloat16)
    x=backing[:,3:3+q+k]
    qw=torch.randn(q,device='cuda',dtype=torch.bfloat16)
    kw=torch.randn(k,device='cuda',dtype=torch.bfloat16)
    qw_before,kw_before=qw.clone(),kw.clone()
    def reference():
        qr,kv=native.fused_q_kv_rmsnorm(x[:,:q],x[:,q:],qw,kw,1e-20)
        if lightop:
            from lightop.quant import per_token_quant_fp8
            data,scale=per_token_quant_fp8(qr)
        else:data,scale=ops.scaled_fp8_quant(qr,use_per_token_if_dynamic=True)
        return data,scale.reshape(m,1),kv
    with set_current_vllm_config(VllmConfig(
            compilation_config=CompilationConfig(mode=0,custom_ops=['all']))):
        if not m:
            values=norm_quantize(x,qw,kw,1e-20,lightop=lightop)
            assert [tuple(v.shape) for v in values]==[(m,q),(m,1),(m,k)]
            return
        x.normal_();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):norm_quantize(x,qw,kw,1e-20,lightop=lightop)
        stream.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            captured=norm_quantize(x,qw,kw,1e-20,lightop=lightop)
        for amp in (0.,1e-30,.25,8.):
            x.normal_(std=amp);before=x.clone();graph.replay()
            for actual,expected in zip(captured,reference()):
                torch.testing.assert_close(actual.view(torch.uint8),expected.view(torch.uint8),rtol=0,atol=0)
            torch.testing.assert_close(x.view(torch.uint8),before.view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(qw,qw_before,rtol=0,atol=0)
        torch.testing.assert_close(kw,kw_before,rtol=0,atol=0)
