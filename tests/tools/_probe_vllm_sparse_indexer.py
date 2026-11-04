"""Bounded V4 indexer public-OOT experiment with real official metadata.

Run from a matching v0.33 environment with HIP_VISIBLE_DEVICES=7,
CUDA_VISIBLE_DEVICES=7, HCU_ADAPTER_HOST=none, VLLM_PLUGINS=hcu_adapter,
HCU_VLLM_SPARSE_INDEXER_IMPL=native and PYTHONPATH='' for an installed wheel
(or this checkout for a source-only check). Use --native-provider --graph;
add --mtp 1 --padding for padded/unequal batches or --mtp 5 for flattening.
Only the selected GPU is used. Capture failures require a fresh process.
This probe uses the real Adapter startup registration and official public metadata.
External diagnostics do not substitute for installed-wheel/lifecycle/model gates.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import traceback

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--mtp', type=int, choices=[1, 5], default=0,
                        help='build the real official MTP config and metadata')
    parser.add_argument('--native-provider', action='store_true',
                        help='experimental C4 metadata orchestration over public official helpers')
    parser.add_argument('--padding', action='store_true', help='real padded and unequal MTP metadata')
    parser.add_argument('--compile', action='store_true', help='real fullgraph dynamic Torch compilation')
    parser.add_argument('--dynamic-lens', action='store_true', help='change padding positions and causal lengths at fixed addresses')
    parser.add_argument('--empty', action='store_true', help='zero-token metadata/Graph control; no numerical work')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    (out / 'driver.py').write_bytes(Path(__file__).read_bytes())
    record = dict(status='RUNNING', cases=[], graph_requested=args.graph,
                  production_changed=True, architecture_overridden=False,
                  registration='isolated diagnostic public CustomOp OOT + unique torch op',
                  full_model_acceptance=False)


    def save():
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')


    try:
        assert os.environ['HIP_VISIBLE_DEVICES'] == os.environ['CUDA_VISIBLE_DEVICES'] == '7'
        assert os.environ['HCU_ADAPTER_HOST'] == 'none'
        import torch
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        from vllm.config import (ModelConfig, CacheConfig, ParallelConfig, VllmConfig,
            CompilationConfig, SchedulerConfig, CUDAGraphMode, SpeculativeConfig,
            set_current_vllm_config)
        from vllm.platforms import current_platform
        from vllm.platforms.rocm import get_cdna_version, on_gfx942, on_gfx950
        from vllm.forward_context import set_forward_context, get_forward_context
        from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
        from vllm.models.deepseek_v4.attention import DeepseekV4IndexerCache
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
        from vllm.v1.attention.ops import rocm_aiter_mla_sparse as official
        from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager, lock_workspace
        from vllm.v1.worker.workspace import current_workspace_manager
        from vllm.utils.torch_utils import direct_register_custom_op
        torch.set_num_threads(1)
        torch.manual_seed(330040)
        arch = torch.cuda.get_device_properties(0).gcnArchName
        assert arch.split(':')[0] == 'gfx936'
        assert torch.cuda.mem_get_info()[0] >= 2 * 1024**3
        record['identity'] = dict(arch=arch, cdna=get_cdna_version(),
            gfx942=on_gfx942(), gfx950=on_gfx950(), rocm=current_platform.is_rocm(),
            cuda=current_platform.is_cuda())
        record['sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in {Path(inspect.getfile(x)) for x in [SparseAttnIndexer,
                DeepseekV4IndexerCache, DeepseekV32IndexerMetadataBuilder, official]}}
        model = ModelConfig(model='/model/DeepSeek-V4-Flash', dtype='bfloat16',
                            max_model_len=8192, language_model_only=True)
        parallel = ParallelConfig(tensor_parallel_size=1)
        speculative = (SpeculativeConfig(method='mtp', num_speculative_tokens=args.mtp,
            target_model_config=model, target_parallel_config=parallel) if args.mtp else None)
        config = VllmConfig(model_config=model,
            parallel_config=parallel, speculative_config=speculative,
            cache_config=CacheConfig(block_size=256),
            scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=128,
                max_model_len=8192, enable_chunked_prefill=True,
                is_encoder_decoder=model.is_encoder_decoder),
            compilation_config=CompilationConfig(custom_ops=['all'],
                cudagraph_capture_sizes=[1, 2, 4], max_cudagraph_capture_size=4,
                inductor_compile_config={'combo_kernels': False}))
        # Layout is resolved after construction in v0.33, not an init argument.
        config.cache_config.kv_cache_layout = 'LBNHC'
        record['config'] = dict(model_architectures=model.architectures,
            max_model_len=model.max_model_len, indexer_heads=64, dim=128, topk=512,
            compress_ratio=4, manager_page=256, indexer_page=64,
            logits_workspace_mb=os.environ.get('VLLM_SPARSE_INDEXER_MAX_LOGITS_MB'),
            mtp_num_speculative_tokens=args.mtp)

        with torch.inference_mode(), set_current_vllm_config(config):
            cache_layer = DeepseekV4IndexerCache(132, torch.uint8,
                'model.layers.2.attn.indexer.k_cache', config.cache_config, compress_ratio=4)
            prefix = cache_layer.prefix
            spec = cache_layer.get_kv_cache_spec(config)
            assert spec.num_states == 64
            builder = DeepseekV32IndexerMetadataBuilder(spec, [prefix], config,
                torch.device('cuda:0'), block_table_width=32)
            record['builder'] = dict(real_constructor=True,
                use_flattening=builder.use_flattening, supports_varlen=builder.supports_varlen,
                decode_threshold=builder.decode_threshold)
            maxlen = 2048
            pages = 110
            base_values = (torch.randn(pages, 64, 128) * .5).to(current_platform.fp8_dtype())
            base_scales = torch.exp2(torch.randint(-2, 2, (pages, 64)).float())
            cache = torch.empty(pages, 64, 132, dtype=torch.uint8, device='cuda')
            unbound_cache = cache_layer.kv_cache
            cache_layer.bind_kv_cache(cache)
            topk = torch.full((128, 512), -7, dtype=torch.int32, device='cuda')
            op = SparseAttnIndexer(cache_layer, 128, 'ue8m0', 512, 128,
                maxlen, 8192, topk, skip_k_cache_insert=True, compress_ratio=4)
            from hcu_adapter.bridges import vllm_sparse_indexer as bridge
            bridge.require_healthy()
            assert type(op).__module__ == bridge.__name__
            record['public_oot_dispatch_selected'] = True
            record['registration'] = 'real Adapter general-plugin startup/public OOT'
            record['bridge_sha256'] = hashlib.sha256(Path(bridge.__file__).read_bytes()).hexdigest()
            record['bridge_file'] = bridge.__file__
            uncompiled = op
            init_workspace_manager(torch.device('cuda:0'))
            cache_layer.kv_cache = unbound_cache
            with set_forward_context(None, config):
                uncompiled(torch.zeros(1,1,device='cuda'),
                    torch.zeros(1,64,128,device='cuda').to(current_platform.fp8_dtype()),
                    None, torch.zeros(1,64,device='cuda'))
            cache_layer.bind_kv_cache(cache)
            record['unbound_cache_profile_pass'] = True
            if args.compile:
                op = torch.compile(op, fullgraph=True, dynamic=True)
            record['fullgraph_compile_requested'] = args.compile
            tests = [('decode', [1, 1], [4105, 2313]),
                     ('prefill', [17, 9], [4105, 2313]),
                     ('mixed', [1, 17], [4105, 2323])]
            if args.mtp:
                tests = [('mtp', [args.mtp+1, args.mtp+1], [4105, 2313])]
            if args.padding:
                tests = [('padded_plain', [1,0], [4104,0]),
                         ('padded_mtp', [2,0], [4104,0]),
                         ('unequal_decode_first', [2,1], [4104,2312]),
                         ('unequal_single_first', [1,2], [4104,2312]),
                         ('padded_middle', [2,0,2], [4104,0,2312]),
                         ('padded_tail', [2,2,0], [4104,2312,0])]
            if args.dynamic_lens:
                assert args.mtp == 1
                tests = [('dynamic_padded', [2,2,0], [4104,2312,0])]
            if args.empty:
                tests = [('empty', [0,0], [0,0])]
            if args.graph and not args.native_provider:
                tests = tests[:1]
            for kind, query_lens, seq_lens in tests:
                reset_workspace_manager()
                init_workspace_manager(torch.device('cuda:0'))
                tokens = sum(query_lens)
                q = (torch.randn(tokens, 64, 128, device='cuda') * .5).to(current_platform.fp8_dtype())
                weights = torch.randn(tokens, 64, device='cuda') * .125
                hidden = torch.zeros(tokens, 1, device='cuda')
                offsets_cpu = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32)
                offsets = offsets_cpu.cuda()
                seq_cpu = torch.tensor(seq_lens, dtype=torch.int32)
                seq = seq_cpu.cuda()
                tables_cpu = torch.randperm(pages)[:len(query_lens)*32].view(len(query_lens), 32).int()
                tables = tables_cpu.cuda()
                slots = torch.full((tokens,), -123, dtype=torch.int64, device='cuda')
                common = CommonAttentionMetadata(query_start_loc=offsets,
                    query_start_loc_cpu=offsets_cpu, seq_lens=seq,
                    seq_lens_cpu_upper_bound=seq_cpu, num_reqs=len(query_lens),
                    num_actual_tokens=tokens, max_query_len=max(query_lens),
                    max_seq_len=max(seq_lens), block_table_tensor=tables,
                    slot_mapping=slots, causal=True)
                metadata = builder.build(0, common)
                case = dict(kind=kind, states=[], num_decodes=metadata.num_decodes,
                    num_prefills=metadata.num_prefills, num_decode_tokens=metadata.num_decode_tokens,
                    num_prefill_tokens=metadata.num_prefill_tokens,
                    compressed_slot_mapping=metadata.slot_mapping.cpu().tolist(),
                    source_q_lens=query_lens, source_seq_lens=seq_lens)
                if metadata.decode is not None:
                    case['decode_metadata'] = dict(seq_lens=metadata.decode.seq_lens.cpu().tolist(),
                        decode_lens=metadata.decode.decode_lens.cpu().tolist(),
                        requires_padding=metadata.decode.requires_padding,
                        block_table_shape=list(metadata.decode.block_table.shape),
                        write_max_decode_len=metadata.decode.write_max_decode_len)
                record['cases'].append(case)
                q_base, w_base = q.clone(), weights.clone()
                graph = None
                tracked = [q, cache, weights, tables, metadata.slot_mapping]
                if metadata.decode is not None:
                    tracked.extend([metadata.decode.seq_lens, metadata.decode.decode_lens])
                if metadata.prefill is not None:
                    for chunk in metadata.prefill.chunks:
                        tracked.extend([chunk.cu_seqlen_ks, chunk.cu_seqlen_ke,
                                        chunk.cu_seq_lens, chunk.token_to_seq])
                addresses = [x.data_ptr() for x in tracked]
                previous_ids = None
                for state in range(3):
                    if args.dynamic_lens:
                        query_lens = [2,0,2] if state == 1 else [2,2,0]
                        seq_lens = [4108,0,2316] if state == 1 else [4104+state*4,2312+state*4,0]
                        offsets_cpu.copy_(torch.tensor([0,*torch.tensor(query_lens).cumsum(0).tolist()],dtype=torch.int32))
                        offsets.copy_(offsets_cpu)
                        seq_cpu.copy_(torch.tensor(seq_lens,dtype=torch.int32))
                        seq.copy_(seq_cpu)
                        common.max_seq_len = max(seq_lens)
                        updated = builder.build(0,common)
                        assert updated.num_decode_tokens == metadata.num_decode_tokens
                        assert updated.decode.requires_padding and updated.prefill is None
                        assert updated.decode.seq_lens.data_ptr() == metadata.decode.seq_lens.data_ptr()
                        assert updated.decode.decode_lens.data_ptr() == metadata.decode.decode_lens.data_ptr()
                        case.setdefault('dynamic_metadata',[]).append(dict(state=state,query_lens=query_lens,seq_lens=seq_lens,compressed_lens=metadata.decode.seq_lens.cpu().tolist()))
                    values = (base_values.float() * (1 + state / 8)).to(current_platform.fp8_dtype())
                    scales = base_scales * (1 + state / 4)
                    flat = cache.view(pages, -1)
                    flat[:, :64*128].copy_(values.reshape(pages, -1).view(torch.uint8))
                    flat[:, 64*128:].copy_(scales.contiguous().view(torch.uint8).view(pages, -1))
                    q.copy_((q_base.float().roll(state, -1) * (1-state/16)).to(q.dtype))
                    weights.copy_(w_base.roll(state, -1) * (1 + state / 8))
                    before = [x.clone() for x in tracked]
                    with set_forward_context(None, config):
                        assert op(hidden, q, None, weights).data_ptr() == topk.data_ptr()
                    with set_forward_context({prefix: metadata}, config,
                            cudagraph_runtime_mode=CUDAGraphMode.FULL if args.graph else CUDAGraphMode.NONE):
                        if args.compile:
                            uncompiled_ids = uncompiled(hidden, q, None, weights).clone()
                        output = op(hidden, q, None, weights)
                        if args.compile:
                            assert torch.equal(output.sort(-1).values, uncompiled_ids.sort(-1).values)
                        if args.graph:
                            eager_ids = output.clone()
                            if graph is None:
                                lock_workspace()
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    captured = op(hidden, q, None, weights)
                            graph.replay()
                            torch.cuda.synchronize()
                            assert torch.equal(captured.sort(-1).values,
                                               eager_ids.sort(-1).values)
                    got = output[:tokens].cpu()
                    assert all(torch.equal(x.contiguous().view(torch.uint8),
                        b.contiguous().view(torch.uint8)) for x, b in zip(tracked, before))
                    assert addresses == [x.data_ptr() for x in tracked]
                    query, w = q.cpu().double(), weights.cpu().double()
                    results = []
                    row = 0
                    for request, (ql, length) in enumerate(zip(query_lens, seq_lens)):
                        n = length // 4
                        physical = tables_cpu[request, torch.arange(n)//64].long()
                        positions = torch.arange(n)%64
                        keys = values[physical, positions].double()
                        scale = scales[physical, positions].double()
                        for j in range(ql):
                            end = (length - ql + j + 1)//4
                            dot = query[row] @ keys.T
                            is_prefill = row >= metadata.num_decode_tokens
                            declared = dot.to(torch.bfloat16).double() if is_prefill else dot
                            gold = (declared.clamp_min(0) * scale[None, :] * w[row, :, None]).sum(0)[:end]
                            pure = (dot.clamp_min(0) * scale[None, :] * w[row, :, None]).sum(0)[:end]
                            selected = got[row][got[row] >= 0].long()
                            assert selected.numel() == min(512, end)
                            assert selected.unique().numel() == selected.numel()
                            assert bool((selected < end).all())
                            expected = gold.topk(min(512, end)).values.sort().values
                            actual = gold[selected].sort().values
                            error = ((expected-actual).abs()/(.002+.002*expected.abs())).max().item()
                            pure_error = ((pure.topk(min(512, end)).values.sort().values-
                                pure[selected].sort().values).abs()/
                                (.002+.002*pure.topk(min(512, end)).values.sort().values.abs())).max().item()
                            results.append(dict(row=row, valid_end=end,
                                selected=selected.numel(), declared_selection_error=error,
                                pure_fp64_selection_error=pure_error))
                            if error > 1:
                                failure = out / 'selection-failure.pt'
                                torch.save(dict(query=q.cpu(), weights=weights.cpu(),
                                    cache=cache.cpu(), topk=got, block_table=tables_cpu,
                                    row=row, reference=gold, pure_fp64=pure), failure)
                                record['failure_artifact'] = dict(path=failure.name,
                                    sha256=hashlib.sha256(failure.read_bytes()).hexdigest())
                                record['selection_failure'] = results[-1]
                            assert error <= 1, results[-1]
                            row += 1
                    case['states'].append(dict(state=state, cache_unchanged=True,
                        all_inputs_unchanged=True, addresses_stable=True, rows=results,
                        graph_replayed=args.graph, compiled_uncompiled_selection_sets_equal=args.compile,
                        changed_selection=previous_ids is None or not torch.equal(previous_ids, got),
                        changed_selection_sets=previous_ids is None or not torch.equal(
                            previous_ids.sort(-1).values, got.sort(-1).values)))
                    if tokens:
                        assert case['states'][-1]['changed_selection_sets']
                    else:
                        assert got.numel() == 0
                        assert torch.equal(topk, torch.full_like(topk, -7))
                        case['states'][-1]['no_numerical_work'] = True
                    previous_ids = got.clone()
                    save()
                case['status'] = 'GRAPH_COMPONENT_PASS' if args.graph else 'EAGER_COMPONENT_PASS'
            status = ('PUBLIC_OOT_NATIVE_C4_CONTEXT_GRAPH_COMPONENT_PASS' if args.graph and args.native_provider
                      else 'PUBLIC_OOT_OFFICIAL_CONTEXT_EAGER_COMPONENT_PASS')
            record.update(status=status,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                remaining=['installed wheel and lifecycle integration', 'FULL engine Graph',
                           'cache insertion/compressor golden', 'layer/model goldens and HumanEval32'])
    except BaseException as error:
        record.update(status='CONTEXT_COMPONENT_HOLD',
            error=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
    finally:
        save()
    print(record['status'], flush=True)
    raise SystemExit(0 if record['status'] in {'PUBLIC_OOT_OFFICIAL_CONTEXT_EAGER_COMPONENT_PASS',
        'PUBLIC_OOT_NATIVE_C4_CONTEXT_GRAPH_COMPONENT_PASS'} else 1)


if __name__ == '__main__':
    main()
