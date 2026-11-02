"""Real Adapter registration -> official FP8 family -> quantizer -> GEMM/Graph.

No model, no forced kernel, no architecture/predicate changes. Prepared official
venv plus this worktree on an explicit isolated PYTHONPATH; HIP_VISIBLE_DEVICES=3.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('HIP_VISIBLE_DEVICES')=='3'
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.device_count()==1
    free,total=torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(4*1024**3,free*.5)/total)
    import vllm
    import hcu_adapter.bridges.vllm as bridge
    from vllm.config import VllmConfig, set_current_vllm_config
    import vllm.model_executor.kernels.linear as linear
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTensorSym,kFp8DynamicTokenSym,kFp8StaticChannelSym,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import _supports_torch_fp8_scaled_mm
    from hcu_adapter import registry
    before=list(linear._POSSIBLE_FP8_KERNELS[linear.PlatformEnum.ROCM])
    native_before=_supports_torch_fp8_scaled_mm()
    bridge.hcu_runtime_plugin()
    after=list(linear._POSSIBLE_FP8_KERNELS[linear.PlatformEnum.ROCM])
    assert after[:len(before)]==before
    assert len(after)==len(before)+1
    assert after[-1].__name__=='TritonFP8ScaledMMLinearKernel'
    assert _supports_torch_fp8_scaled_mm()==native_before==False
    torch.manual_seed(20261031)
    records=[]
    with set_current_vllm_config(VllmConfig()):
        for m,k,n in [(1,512,256),(17,512,256),(17,2560,2560)]:
            for key in [kFp8DynamicTensorSym,kFp8DynamicTokenSym]:
                for dtype in [torch.bfloat16,torch.float16]:
                    record=dict(shape=[m,k,n],quant_key=str(key),dtype=str(dtype),passed=False)
                    try:
                        kernel=linear.init_fp8_linear_kernel(key,kFp8StaticChannelSym,dtype,dtype,(n,k))
                        assert type(kernel) is after[-1]
                        record['selected_class']=type(kernel).__module__+'.'+type(kernel).__name__
                        x=torch.randn(m,k,device='cuda',dtype=dtype)*64
                        w=torch.randn(n,k,device='cuda').to(torch.float8_e4m3fn).t()
                        bs=torch.linspace(.125,.875,n,device='cuda').reshape(n,1)
                        bias=torch.randn(n,device='cuda',dtype=dtype) if m>1 else None
                        layer=SimpleNamespace(weight=w,weight_scale=bs,input_scale=None,input_scale_ub=None)
                        kernel.process_weights_after_loading(layer)
                        def run():return kernel.apply_weights(layer,x,bias)
                        def check(out):
                            # Reuse the actual official quantizer; reference GEMM/scales/bias are FP32.
                            q,sa=kernel.quant_fp8(x,None,None)
                            ref=(q.float()@w.float())*sa*bs.t()
                            if bias is not None:ref+=bias.float()
                            delta=out.float()-ref
                            cosine=torch.nn.functional.cosine_similarity(out.float().flatten(),ref.flatten(),dim=0).item()
                            nrmse=(delta.square().mean()/ref.square().mean().clamp_min(1e-20)).sqrt().item()
                            relative=(delta.abs().max()/ref.abs().max().clamp_min(1e-20)).item()
                            return dict(cosine=cosine,nrmse=nrmse,relative_to_ref_max_abs=relative,
                                activation_scale_range=[sa.min().item(),sa.max().item()],
                                passed=bool(torch.isfinite(out).all().item() and cosine>=.9999 and nrmse<=.01 and relative<=.01))
                        rows=[dict(stage='eager',**check(run()))]
                        stream=torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(2):run()
                        torch.cuda.current_stream().wait_stream(stream)
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):out=run()
                        pointers=[t.data_ptr() for t in [x,w,bs,out]]
                        previous=None
                        for wave in range(3):
                            x.copy_(torch.randn_like(x)*48)
                            w.copy_(torch.randn_like(w.float()).to(w.dtype))
                            bs.copy_(torch.rand_like(bs)+.15)
                            if bias is not None:bias.copy_(torch.randn_like(bias))
                            graph.replay()
                            torch.cuda.synchronize()
                            row=dict(stage=f'graph-{wave}',**check(out))
                            row['same_addresses']=pointers==[t.data_ptr() for t in [x,w,bs,out]]
                            row['changed']=previous is None or not torch.equal(previous,out)
                            row['passed']=row['passed'] and row['same_addresses'] and row['changed']
                            previous=out.clone()
                            rows.append(row)
                        record.update(results=rows,passed=all(r['passed'] for r in rows))
                        del graph
                    except Exception:record['error']=traceback.format_exc()
                    records.append(record)
                    with (args.output/'cases.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
                    print(json.dumps(record),flush=True)
    (args.output/'protocol.json').write_text(json.dumps(dict(torch=torch.__version__,hip=torch.version.hip,
        vllm=vllm.__version__,arch=torch.cuda.get_device_properties(0).gcnArchName,
        bridge_file=bridge.__file__,before=[c.__name__ for c in before],after=[c.__name__ for c in after],
        native_support_unchanged=native_before==_supports_torch_fp8_scaled_mm()==False,
        registry=registry.records(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        thresholds=dict(cosine_min=.9999,nrmse_max=.01,relative_to_ref_max_abs_max=.01)),indent=2,sort_keys=True)+'\n')
    (args.output/Path(__file__).name).write_text(Path(__file__).read_text())
    (args.output/'summary.json').write_text(json.dumps(dict(cases=len(records),passed=sum(r['passed'] for r in records),
        all_passed=all(r['passed'] for r in records),peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved()),indent=2)+'\n')
    (args.output/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n'
        for p in sorted(args.output.iterdir()) if p.is_file()))
    raise SystemExit(0 if all(r['passed'] for r in records) else 1)


if __name__=='__main__':main()
