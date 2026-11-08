# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bounded installed-vLLM Linear replay; write JSON artifacts, never a service.

Run with one visible idle device, HCU_ADAPTER_HOST=none, VLLM_PLUGINS=hcu_adapter,
Adapter development PYTHONPATH, --output a fresh directory. Written before the
LightOp integration. Does not replace platform or quantization implementations.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import vllm
    from hcu_adapter.bridges.vllm_fp8 import install as install_fp8
    from hcu_adapter.bridges.vllm_scaled_mm import install
    from hcu_adapter.kernels.scaled_mm_lightop import supported
    from vllm.model_executor.kernels import linear
    from vllm.model_executor.kernels.linear.scaled_mm import triton as native
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym, kFp8StaticChannelSym,
    )
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    from vllm.config import VllmConfig, set_current_vllm_config
    context = set_current_vllm_config(VllmConfig())
    context.__enter__()
    install_fp8()
    first = install()
    assert install() is False
    protocol = dict(torch=torch.__version__, vllm=vllm.__version__, hip=torch.version.hip,
                    architecture=arch, first_install=first, source_files={})
    for file in [Path(__file__), Path(inspect.getfile(install)),
                 Path(inspect.getfile(supported)), Path(native.__file__)]:
        protocol['source_files'][str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (args.output / Path(__file__).name).write_text(Path(__file__).read_text())
    records = []

    def metrics(out, ref):
        assert out.shape == ref.shape and out.device == ref.device
        if not out.numel():
            return dict(cosine=1., relative_l2=0., normalized_max=0.)
        x, y = out.float().flatten(), ref.float().flatten()
        delta = x - y
        cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        rel = (delta.norm() / y.norm().clamp_min(1e-20)).item()
        mx = (delta.abs().max() / y.abs().max().clamp_min(1e-20)).item()
        assert torch.isfinite(x).all() and cos >= .9999 and rel <= .01 and mx <= .01, (cos, rel, mx)
        return dict(cosine=cos, relative_l2=rel, normalized_max=mx)

    def save(record):
        records.append(record)
        with (args.output / 'cases.jsonl').open('a') as f:
            f.write(json.dumps(record, sort_keys=True) + '\n')
        if not record['passed']:
            print(json.dumps(record), flush=True)

    for kind in ['int8', 'fp8']:
        dt = torch.int8 if kind == 'int8' else torch.float8_e4m3fn
        for seed in range(200):
            row = dict(kind=kind, seed=seed, passed=False)
            try:
                torch.manual_seed(50800 + seed)
                m = [1, 2, 7, 17, 65][seed % 5]
                n = [32, 64, 128][seed % 3]
                k = [32, 128, 256][(seed // 3) % 3]
                od = [torch.bfloat16, torch.float16][seed % 2]
                A = (torch.randn(m, k, device='cuda') * 5).to(dt)
                B = (torch.randn(n, k, device='cuda') * 5).to(dt).t()
                As = (torch.rand(m if seed % 3 else 1, 1, device='cuda') + .1)
                Bs = (torch.rand(n if seed % 4 else 1, 1, device='cuda') + .1)
                bias = torch.randn(n, device='cuda', dtype=od) if seed % 2 else None
                row.update(shape=[m,k,n], vendor=supported(A,B,As,Bs,od,bias,arch))
                if kind == 'int8':
                    call = lambda: torch.ops.vllm.int8_scaled_mm_func(A,B,As,Bs,od,bias)
                else:
                    p = object.__new__(native.TritonFP8ScaledMMLinearKernel)
                    call = lambda: p.apply_scaled_mm(A=A,B=B,As=As,Bs=Bs,out_dtype=od,bias=bias,output_shape=[m,n])
                def reference():
                    r = (A.float() @ B.float()) * As * Bs.t()
                    return r if bias is None else r + bias.float()
                out = call()
                assert out.dtype == od
                row['eager'] = metrics(out, reference())
                if seed < 12:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            call()
                    torch.cuda.current_stream().wait_stream(stream)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        result = call()
                    g.replay()
                    torch.cuda.synchronize()
                    metrics(result, reference())
                    ptrs = [t.data_ptr() for t in (A,B,As,Bs,result)]
                    row['graph'] = []
                    for wave in range(3):
                        A.copy_((torch.randn(m,k,device='cuda') * 5).to(dt))
                        B.copy_((torch.randn(n,k,device='cuda') * 5).to(dt).t())
                        As.copy_(torch.rand_like(As)+.1)
                        Bs.copy_(torch.rand_like(Bs)+.1)
                        if bias is not None:
                            bias.copy_(torch.randn_like(bias))
                        g.replay()
                        torch.cuda.synchronize()
                        assert ptrs == [t.data_ptr() for t in (A,B,As,Bs,result)]
                        row['graph'].append(metrics(result, reference()))
                    del g
                row['passed'] = True
            except Exception:
                row['error'] = traceback.format_exc()
            save(row)
        # Exercise native selector, post-load and native quantization/forward.
        for m in [0,1,17]:
            row = dict(kind=kind, lifecycle=True, m=m, passed=False)
            try:
                od = torch.bfloat16
                n,k = 64,128
                layer = torch.nn.Module()
                layer.logical_widths = [n]
                w = (torch.randn(n,k,device='cuda')*5).to(dt)
                layer.weight = torch.nn.Parameter(w if kind=='int8' else w.t(), requires_grad=False)
                layer.weight_scale = torch.nn.Parameter(torch.rand(n,1,device='cuda')+.1, requires_grad=False)
                layer.input_scale = None
                if kind=='int8':
                    p = linear.init_int8_linear_kernel(True,False,True,'probe')
                else:
                    p = linear.init_fp8_linear_kernel(kFp8DynamicTokenSym,kFp8StaticChannelSym,od,od,(n,k))
                p.process_weights_after_loading(layer)
                x = torch.randn(m,k,device='cuda',dtype=od)
                bias = torch.randn(n,device='cuda',dtype=od)
                if m == 0:
                    try:
                        p.apply_weights(layer,x,bias)
                    except AssertionError:
                        row.update(passed=True, expected_refusal='upstream M > 0 contract', provider=type(p).__name__)
                    else:
                        raise RuntimeError('Expected upstream empty-input refusal')
                    save(row)
                    continue
                y = p.apply_weights(layer,x,bias)
                # Independent dot after native quantization, not after vendor execution.
                if kind=='int8':
                    from vllm import _custom_ops as ops
                    q,s,_ = ops.scaled_int8_quant(x.contiguous(),None,None,symmetric=True)
                else:
                    q,s = p.quant_fp8(x,None,None)
                ref = (q.float() @ layer.weight.float()) * s * layer.weight_scale.t() + bias.float()
                row.update(provider=type(p).__name__,metrics=metrics(y,ref),passed=True)
            except Exception:
                row['error'] = traceback.format_exc()
            save(row)
    # Exercise metadata fallbacks without fabricating a different architecture.
    for mode in ['empty','unaligned','rowmajor','rowscale']:
        row = dict(fallback=mode,passed=False)
        try:
            m,k,n = (0,32,64) if mode=='empty' else (7,33,65) if mode=='unaligned' else (7,32,64)
            A = torch.randint(-8,8,(m,k),device='cuda',dtype=torch.int8)
            B = torch.randint(-8,8,(n,k),device='cuda',dtype=torch.int8).t()
            if mode=='rowmajor':
                B = B.contiguous()
            As = torch.full((m,1),.3,device='cuda')
            Bs = torch.full((1,n) if mode=='rowscale' else (n,1),.7,device='cuda')
            assert not supported(A,B,As,Bs,torch.bfloat16,None,arch)
            if mode in ('empty', 'rowscale'):
                try:
                    native.triton_scaled_mm(A,B,As,Bs,torch.bfloat16)
                except AssertionError:
                    row.update(passed=True, expected_refusal='upstream shape contract')
                else:
                    raise RuntimeError('Expected upstream shape refusal')
                save(row)
                continue
            y = native.triton_scaled_mm(A,B,As,Bs,torch.bfloat16)
            ref = (A.float() @ B.float()) * As * Bs.reshape(1,n)
            row.update(metrics=metrics(y,ref),passed=True)
        except Exception:
            row['error'] = traceback.format_exc()
        save(row)
    summary = dict(cases=len(records),passed=sum(r['passed'] for r in records),
                   vendor_cases=sum(r.get('vendor',False) for r in records),
                   all_passed=all(r['passed'] for r in records),architecture=arch,
                   model_validated=False,performance_validated=False)
    (args.output / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    (args.output / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n'
        for p in sorted(args.output.iterdir()) if p.is_file()))
    return 0 if summary['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
