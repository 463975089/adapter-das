# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real builder/DTK component; no model weights, EP communication or PD serving."""
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


def load_config():
    model = Path(os.environ.get('HCU_HY4_MODEL_DIR', '/nonexistent-hy4-model'))
    if not torch.cuda.is_available() or torch.version.hip is None or not model.is_dir():
        pytest.skip('requires HCU and HCU_HY4_MODEL_DIR')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] not in ('gfx936', 'gfx938'):
        pytest.skip('requires gfx936/gfx938')
    from hcu_adapter.bridges.vllm import hcu_runtime_plugin
    hcu_runtime_plugin()
    from vllm.entrypoints.common.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.engine.arg_utils import AsyncEngineArgs
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args([str(model), '--trust-remote-code',
        '--max-model-len', '89088', '--max-num-batched-tokens', '32',
        '--max-num-seqs', '8', '--block-size', '64', '--tensor-parallel-size', '1',
        '--data-parallel-size', '16', '--data-parallel-size-local', '16',
        '--enable-expert-parallel', '--all2all-backend', 'deepep_low_latency',
        '--moe-backend', 'deep_gemm', '--attention-backend', 'FLASHMLA_SPARSE',
        '--kv-cache-dtype', 'fp8_e4m3',
        '--speculative-config', '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"FLASHMLA_SPARSE"}',
        '--compilation-config', '{"mode":0,"cudagraph_mode":"FULL"}'])
    args.model = args.model_tag
    cfg = AsyncEngineArgs.from_cli_args(args).create_engine_config()
    assert cfg.cache_config.cache_dtype == 'fp8_ds_mla'
    assert cfg.parallel_config.prefill_context_parallel_size == 1
    return cfg


def candidate_module():
    source = os.environ.get('HCU_SPARSE_DRAFT_SOURCE')
    if source:
        # Source-stage diagnostic only. Installed qualification does not set it.
        spec = importlib.util.spec_from_file_location('hy4_sparse_draft_candidate', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from hcu_adapter.kernels import vllm_sparse_flashmla_dtk
    return vllm_sparse_flashmla_dtk


class DraftCase:
    """Two actual draft attention steps with stable buffers and native builders."""

    rows, page, pages, topk = 8, 64, 128, 2048

    def __init__(self, config, tp=1, sink=True):
        from flash_mla import flash_mla_interface as vendor
        from hcu_adapter.infra.native_namespace import bind_native
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        from vllm.v1.attention.backends.mla.flashmla_sparse import (
            FlashMLASparseMetadataBuilder as NativeBuilder,
        )
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerMetadataBuilder as IndexerBuilder,
            DeepseekV32IndexerBackend,
        )
        from vllm.v1.worker.utils import AttentionGroup, customize_attention_spec
        from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

        self.cfg = cfg = copy.copy(config)
        cfg.parallel_config = copy.copy(config.parallel_config)
        cfg.parallel_config.tensor_parallel_size = tp
        cfg.compilation_config = copy.copy(config.compilation_config)
        cfg.compilation_config.static_forward_context = {
            'draft.attn': NS(prefill_backend=None)}
        self.heads = cfg.model_config.get_num_attention_heads(cfg.parallel_config)
        self.device = torch.device('cuda:0')
        self.owner = candidate_module()
        self.backend = self.owner.build_backend(vendor)

        class ReferenceBuilder(NativeBuilder):
            # Same native body and DTK bindings as the pre-candidate Adapter.
            _build_fp8_mixed_decode_prefill = bind_native(
                NativeBuilder._build_fp8_mixed_decode_prefill,
                {'get_mla_metadata': vendor.get_mla_metadata})
            _build_fp8_separate_prefill_decode = bind_native(
                NativeBuilder._build_fp8_separate_prefill_decode,
                {'get_mla_metadata': vendor.get_mla_metadata})

        hf = cfg.model_config.hf_text_config
        shell = NS(kv_cache_dtype='fp8_ds_mla',
            head_size=hf.kv_lora_rank + hf.qk_rope_head_dim, sliding_window=None,
            topk_buffer_width=hf.index_topk, indexer=object(),
            non_causal_multi_token_decode=False)
        mla_spec = customize_attention_spec(self.backend,
            MLAAttention.get_kv_cache_spec(shell, cfg))
        ishell = NS(cache_config=cfg.cache_config, dtype=torch.uint8,
            head_dim=hf.index_head_dim + hf.index_head_dim // 128 * 4)
        index_spec = customize_attention_spec(DeepseekV32IndexerBackend,
            DeepseekV32IndexerCache.get_kv_cache_spec(ishell, cfg))
        self.width = (cfg.model_config.max_model_len + self.page - 1) // self.page
        with set_current_vllm_config(cfg):
            self.builder = self.backend.get_builder_cls()(
                mla_spec, ['draft.attn'], cfg, self.device)
            self.reference = ReferenceBuilder(mla_spec, ['draft.attn'], cfg, self.device)
            self.indexer = IndexerBuilder(index_spec, ['draft.indexer'], cfg,
                self.device, block_table_width=self.width)
            self.ref_indexer = IndexerBuilder(index_spec, ['draft.indexer'], cfg,
                self.device, block_table_width=self.width)
        assert self.builder.supports_draft_decode_metadata_update
        self.groups = [[
            AttentionGroup(self.backend, ['draft.attn'], mla_spec, 0,
                metadata_builders=[self.builder]),
            AttentionGroup(DeepseekV32IndexerBackend, ['draft.indexer'], index_spec, 0,
                metadata_builders=[self.indexer]),
        ]]
        self.speculator_shell = NS(attn_groups=self.groups, block_tables=NS(cp_size=1))
        DraftModelSpeculator._configure_decode_metadata_update(self.speculator_shell)
        assert self.speculator_shell.supports_decode_metadata_update

        n, h = self.rows, self.heads
        self.qsl = torch.arange(n + 1, dtype=torch.int32, device=self.device)
        self.seq = torch.full((n,), 128, dtype=torch.int32, device=self.device)
        self.table = torch.zeros((n, self.width), dtype=torch.int32, device=self.device)
        self.slots = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        self.q = torch.randn(2, n, h, 576, dtype=torch.bfloat16, device=self.device) * .1
        self.latent = torch.randn(2, n, 512, dtype=torch.bfloat16, device=self.device) * .1
        self.rope = torch.randn(2, n, 64, dtype=torch.bfloat16, device=self.device) * .1
        self.seq_steps = torch.empty((2, n), dtype=torch.int32, device=self.device)
        self.slot_steps = torch.empty((2, n), dtype=torch.int64, device=self.device)
        self.logical = torch.empty((2, n, self.topk), dtype=torch.int32, device=self.device)
        self.cache = torch.zeros((self.pages, self.page, 656), dtype=torch.uint8, device=self.device)
        self.scale = torch.ones(1, device=self.device)
        from vllm import _custom_ops as ops
        total = self.pages * self.page
        ops.concat_and_cache_mla(
            (torch.randn(total, 512, device=self.device) * .1).bfloat16(),
            (torch.randn(total, 64, device=self.device) * .1).bfloat16(), self.cache,
            torch.arange(total, device=self.device, dtype=torch.int64),
            'fp8_ds_mla', self.scale)
        self.impl = object.__new__(self.backend.get_impl_cls())
        self.impl.num_heads = h
        self.impl.fp8_decode_padded_heads = self.builder.fp8_decode_padded_heads
        self.impl.softmax_scale = .25
        self.impl.sinks = torch.linspace(-1, 1, h, device=self.device) if sink else None
        self.stage(n, 0)
        self.seq.copy_(self.seq_steps[0]); self.slots.copy_(self.slot_steps[0])
        common = self.common(n)
        self.md = self.builder.build(0, common)
        self.imd = self.indexer.build(0, self.common(n))
        self.rmd = self.reference.build(0, self.common(n))
        self.metadata = {'draft.attn': self.md, 'draft.indexer': self.imd}
        self.addresses = self.pointers()

    def common(self, active):
        from vllm.v1.attention.backend import CommonAttentionMetadata
        # Native sparse splitting has one collective placeholder request when
        # there are no actual tokens; a zero-request fixture is not its API.
        requests = max(active, 1)
        return CommonAttentionMetadata(query_start_loc=self.qsl[:requests + 1],
            query_start_loc_cpu=torch.arange(requests + 1, dtype=torch.int32).clamp(max=active),
            seq_lens=self.seq[:requests], num_reqs=requests, num_actual_tokens=active,
            max_query_len=1, max_seq_len=self.cfg.model_config.max_model_len,
            block_table_tensor=self.table[:requests], slot_mapping=self.slots[:active],
            seq_lens_cpu_upper_bound=self.seq[:requests].cpu())

    def pointers(self):
        return tuple(x.data_ptr() for x in [self.md.query_start_loc, self.md.seq_lens,
            self.md.slot_mapping, self.md.block_table, self.md.req_id_per_token,
            self.imd.decode.seq_lens, self.imd.decode.block_table])

    def stage(self, active, generation):
        n = self.rows
        qsl = torch.arange(n + 1, dtype=torch.int32).clamp(max=active)
        self.qsl.copy_(qsl)
        # Repeated physical pages keep this a small metadata/attention fixture.
        # The table is fully valid; this does not model an allocator or PD transfer.
        table = (torch.arange(n * self.width).reshape(n, self.width) * 17
                 + generation * 31) % self.pages
        self.table.copy_(table)
        base = torch.tensor([63, 64, 65, 127, 128, 129, 85000, 89080], dtype=torch.int32)
        base += generation % 4 - (3 if generation == 2 else 0)
        for step in range(2):
            lens = base + step
            lens[active:] = 0
            pos = (lens - 1).clamp(min=0).long()
            slots = table[torch.arange(n), pos // self.page] * self.page + pos % self.page
            slots[active:] = -1
            ids = (torch.arange(self.topk)[None] * lens[:, None] // self.topk).int()
            if generation % 2:
                # A reused MTP top-k buffer may retain valid old indices on
                # padded rows. They must remain harmless even without clearing.
                ids[lens == 0] = torch.arange(self.topk, dtype=torch.int32)
            else:
                ids[lens == 0] = -1
            self.seq_steps[step].copy_(lens)
            self.slot_steps[step].copy_(slots)
            self.logical[step].copy_(ids)
        self.q.copy_(torch.randn_like(self.q) * .1)
        self.latent.copy_(torch.randn_like(self.latent) * .1)
        self.rope.copy_(torch.randn_like(self.rope) * .1)
        if self.impl.sinks is not None:
            self.impl.sinks.add_(.05)

    def step(self, step, md, update=False, stage_inputs=True):
        from vllm import _custom_ops as ops
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index,
        )
        if stage_inputs:
            self.seq.copy_(self.seq_steps[step]); self.slots.copy_(self.slot_steps[step])
        if update:
            from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
            DraftModelSpeculator._update_draft_decode_metadata(
                self.speculator_shell, self.metadata, self.rows)
        ops.concat_and_cache_mla(self.latent[step], self.rope[step], self.cache,
            self.slots, 'fp8_ds_mla', self.scale)
        ids = triton_convert_req_index_to_global_index(md.req_id_per_token,
            md.block_table, self.logical[step], BLOCK_SIZE=self.page,
            NUM_TOPK_TOKENS=self.topk)
        if md.fp8_use_mixed_batch:
            q, ids, km = self.q[step].unsqueeze(0), ids.unsqueeze(0), md.fp8_extra_metadata
        else:
            q, ids = self.q[step].unsqueeze(1), ids.unsqueeze(1)
            km = md.fp8_extra_metadata.decode.kernel_metadata
        result, _ = self.impl._fp8_flash_mla_kernel(q, self.cache, ids, km)
        return result.reshape(self.rows, self.heads, 512)


@pytest.fixture(scope='module')
def config():
    return load_config()


@pytest.mark.parametrize('tp,sink', [(1, True), (1, False), (2, True), (8, True)])
@torch.inference_mode()
def test_real_builders_changed_input_fused_graph(config, tp, sink):
    from vllm.config import set_current_vllm_config
    with set_current_vllm_config(config):
        case = DraftCase(config, tp, sink)
    with set_current_vllm_config(case.cfg):
        # Warm actual DTK scheduler and the native indexer update dependency.
        for _ in range(3):
            for step in range(2):
                case.step(step, case.md, True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [case.step(step, case.md, True) for step in range(2)]
        for generation, active in enumerate([8, 3, 1, 0, 8]):
            case.stage(active, generation)
            original_cache = case.cache.clone()
            expected = []
            for step in range(2):
                case.seq.copy_(case.seq_steps[step]); case.slots.copy_(case.slot_steps[step])
                # The runtime rebuild updates native builder buffers; the old
                # single-step Graph retains its captured shape/metadata object.
                fresh = case.reference.build(0, case.common(active))
                assert fresh.num_actual_tokens == active
                result = case.step(step, case.rmd).clone()
                if active == case.rows:
                    fresh_result = case.step(step, fresh)
                    assert torch.equal(result, fresh_result), 'fresh scheduler differs'
                expected.append(result)
            case.cache.copy_(original_cache)
            graph.replay()
            torch.cuda.synchronize()
            assert case.pointers() == case.addresses
            for actual, ref in zip(outputs, expected):
                assert torch.equal(actual[:active], ref[:active])
                assert torch.isfinite(actual[:active]).all()
            assert torch.equal(case.md.req_id_per_token[:active],
                torch.arange(active, device=case.device, dtype=torch.int32))
            if active:
                fresh_index = case.ref_indexer.build(0, case.common(active))
                assert torch.equal(case.imd.decode.seq_lens.view(-1)[:active],
                    fresh_index.decode.seq_lens.view(-1)[:active])
                assert torch.equal(case.imd.decode.block_table[:active],
                    fresh_index.decode.block_table[:active])
            if active == 0:
                assert torch.equal(case.cache, original_cache), 'inactive slots wrote KV'
