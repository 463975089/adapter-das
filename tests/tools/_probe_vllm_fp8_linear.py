"""Bounded real official FP8 provider/changed-input Graph validation, no model.

Run with HIP_VISIBLE_DEVICES=3, PYTHONPATH empty, prepared official vLLM venv.
No support predicate/architecture modifications. The provider is imported from
vLLM; the sole reference is FP32 matmul of decoded FP8 with nonunit scales/bias.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES')!='3':
        parser.error('requires HIP_VISIBLE_DEVICES=3')
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(4*1024**3,free*.5)/total)
    import vllm
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.compressed_tensors import triton_scaled_mm as module
    provider=object.__new__(TritonFP8ScaledMMLinearKernel)
    save(a.output/'protocol.json',dict(torch=torch.__version__,hip=torch.version.hip,
        vllm=vllm.__version__,arch=torch.cuda.get_device_properties(0).gcnArchName,
        gpu=torch.cuda.get_device_name(0),visible_devices=os.environ.get('HIP_VISIBLE_DEVICES'),
        script_sha256=digest(__file__),family_file=inspect.getfile(TritonFP8ScaledMMLinearKernel),
        family_sha256=digest(inspect.getfile(TritonFP8ScaledMMLinearKernel)),
        provider_file=module.__file__,provider_sha256=digest(module.__file__),
        capability_api=hasattr(torch.cuda,'is_scaled_mm_supported'),
        allocator_limit_bytes=4*1024**3,free_start=free,
        thresholds=dict(cosine_min=.9999,nrmse_max=.01,relative_to_ref_max_abs_max=.01),
        note='Direct official provider, no platform predicate changes, model or API. Bias added in FP32 reference.'))
    (a.output/Path(__file__).name).write_text(Path(__file__).read_text())
    torch.manual_seed(20261031)
    x=torch.randn(16,128,device='cuda').to(torch.float8_e4m3fn)
    w=torch.randn(128,128,device='cuda').t().to(torch.float8_e4m3fn)
    native={}
    for mode,sa,sb in [('tensor',torch.full((1,),.5,device='cuda'),torch.full((1,),.75,device='cuda')),
            ('rowwise',torch.full((16,1),.5,device='cuda'),torch.full((1,128),.75,device='cuda'))]:
        try:
            torch._scaled_mm(x,w,scale_a=sa,scale_b=sb,out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            native[mode]=dict(succeeded=True)
        except Exception as exc:
            native[mode]=dict(succeeded=False,error=f'{type(exc).__name__}: {exc}')
    save(a.output/'native.json',native)
    def metrics(out,ref):
        out=out.float()
        delta=out-ref
        cosine=torch.nn.functional.cosine_similarity(out.flatten(),ref.flatten(),dim=0).item()
        nrmse=(delta.square().mean()/ref.square().mean().clamp_min(1e-20)).sqrt().item()
        relative=(delta.abs().max()/ref.abs().max().clamp_min(1e-20)).item()
        return dict(cosine=cosine,nrmse=nrmse,max_abs=delta.abs().max().item(),
            relative_to_ref_max_abs=relative,passed=bool(torch.isfinite(out).all().item()
                and cosine>=.9999 and nrmse<=.01 and relative<=.01))
    cases=[]
    for m,k,n,mode in [(1,128,256,'tensor'),(17,256,128,'token'),(65,512,256,'channel')]:
        for dtype in [torch.bfloat16,torch.float16]:
            for has_bias in [False,True]:
                record=dict(shape=[m,k,n],mode=mode,out_dtype=str(dtype),bias=has_bias,passed=False)
                try:
                    x=torch.randn(m,k,device='cuda').to(torch.float8_e4m3fn)
                    w=torch.randn(n,k,device='cuda').to(torch.float8_e4m3fn).t()
                    sa=torch.linspace(.25,1.25,m if mode=='token' else 1,device='cuda').reshape(-1,1)
                    sb=torch.linspace(.125,.875,1 if mode=='tensor' else n,device='cuda').reshape(-1,1)
                    bias=torch.randn(n,device='cuda',dtype=dtype) if has_bias else None
                    out=None
                    def call():
                        return provider.apply_scaled_mm(A=x,B=w,As=sa,Bs=sb,bias=bias,
                            out_dtype=dtype,output_shape=[m,n])
                    def reference():
                        ref=(x.float()@w.float())*sa*sb.t()
                        return ref if bias is None else ref+bias.float()
                    eager=call()
                    rows=[dict(stage='eager',**metrics(eager,reference()))]
                    stream=torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(2):call()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out=call()
                    pointers=[t.data_ptr() for t in [x,w,sa,sb,out]]
                    previous=None
                    for wave in range(3):
                        x.copy_(torch.randn_like(x.float()).to(x.dtype))
                        w.copy_(torch.randn_like(w.float()).to(w.dtype))
                        sa.copy_(torch.rand_like(sa)+.2)
                        sb.copy_(torch.rand_like(sb)+.15)
                        if bias is not None:bias.copy_(torch.randn_like(bias))
                        graph.replay()
                        torch.cuda.synchronize()
                        row=dict(stage=f'graph-{wave}',**metrics(out,reference()))
                        row['same_addresses']=pointers==[t.data_ptr() for t in [x,w,sa,sb,out]]
                        row['output_changed']=previous is None or not torch.equal(out,previous)
                        row['passed']=row['passed'] and row['same_addresses'] and row['output_changed']
                        previous=out.clone()
                        rows.append(row)
                    record.update(results=rows,passed=all(r['passed'] for r in rows))
                    del graph
                except Exception:
                    record['error']=traceback.format_exc()
                cases.append(record)
                with (a.output/'cases.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
                print(json.dumps(record),flush=True)
    save(a.output/'summary.json',dict(cases=len(cases),passed=sum(c['passed'] for c in cases),
        all_passed=all(c['passed'] for c in cases),peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved()))
    (a.output/'SHA256SUMS').write_text(''.join(digest(p)+'  '+p.name+'\n'
        for p in sorted(a.output.iterdir()) if p.is_file()))
    raise SystemExit(0 if all(c['passed'] for c in cases) else 1)


if __name__=='__main__':
    main()
