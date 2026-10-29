"""Explicit installation at the upstream CUDA backend package seam.

The official ops dispatcher and interface modules execute unchanged. This is
an alternative factory package, not a fake TileLang DSL or an NPU backend.
"""

import importlib.util
from pathlib import Path
import sys

from . import hcu_env
from hcu_adapter import registry
from hcu_adapter.ops.deepinfer_catalog import BACKEND_CATALOG
from hcu_adapter.ops import selector

_BACKEND_PACKAGE = "deepinfer.model_executor.layers.tilelang_ops.cuda"
_DISPATCHER = "deepinfer.model_executor.layers.tilelang_ops.ops"
_PROVIDED = ("gather_tl", "norm_tl", "expand_swa_index_tl",
             "fused_norm_rope_hadamard_cast_store_kvcache_tl", "load_paged_kv_cache_tl",
             "store_kv_cache_tl", "compress_and_update_state_prefill_tl",
             "compress_and_update_state_decode_tl", "kv_index_cat_tl", "convert_kv_index_tl",
             "candidate_block_max_tl", "get_candidate_compact_lens_tl",
             "restore_candidate_indices_inplace_tl", "fused_sample_tl", "random_sampling_tl",
             "prepare_v4_decode_metadata_tl", "copy_block_tables_tl", "gather_input_ids_zos_tl",
             "gather_engram_tokens_zos_tl", "compute_seqlens_k_zos_tl", "load_kv_index_tl",
             "prepare_verification_metadata_tl", "expand_sampling_params_tl", "map_lookup_tl",
             "remap_draft_topk_tl", "get_accept_result_tl", "finalize_verification_tl", "sum_smaller_probs_tl",
             "confidence_head_fused_tl", "dflash_input_store_tl", "copy_dflash_decode_input_tl",
             "store_mtp_info_tl", "compute_seqlens_q_by_confidence_rates_tl", "get_fused_dflash_get_input_tl",
             "select_state_slice_tl", "load_state_to_cache_tl", "apply_token_bitmask_by_step_inplace_tl",
             "detect_outlier_and_abort_tl", "batch_downsample_tl", "prepare_cp_mirror_tl", "replace_input_embeds_tl", "rgb_pad_to_patches_tl", "yoco_cp_boundary_fused_tl")
_backend = None
_norm_backend = None
_selected = {}


def install(backend="triton"):
    """Select Triton or torch explicitly, before importing upstream ops."""
    global _backend, _norm_backend
    if backend not in ("triton", "torch"):
        raise ValueError("Expected backend='triton' or 'torch'")
    if _backend is not None:
        if backend != _backend:
            raise RuntimeError("Changing providers after installation is unsupported; use a fresh process")
        return
    if _DISPATCHER in sys.modules or _BACKEND_PACKAGE in sys.modules:
        raise RuntimeError("Install the HCU backend before importing upstream tilelang_ops.ops")
    path = Path(__file__).with_name("tilelang_backend")
    spec = importlib.util.spec_from_file_location(
        _BACKEND_PACKAGE, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[_BACKEND_PACKAGE] = module
    _backend = backend
    _norm_backend = 'lightop' if hcu_env.lightop_ops_enabled() else backend
    for name in _PROVIDED:
        selected = (_norm_backend if name in ('norm_tl', 'fused_norm_rope_hadamard_cast_store_kvcache_tl')
                    else backend)
        selected = selector.select("tilelang_ops." + name, selected)
        _selected[name] = selected
        registry.register(
            "tilelang_ops." + name, BACKEND_CATALOG["tilelang_ops." + name][selected]["implementation"],
            _DISPATCHER + "." + name, "seam")
        print("HCU adapter runtime: tilelang_ops." + name + " -> " + selected + " (candidate)")


def gather_tl(D, dtype, index_dtype, x_stride, threads=256):
    _backend = _selected['gather_tl']
    from hcu_adapter.kernels.gather import gather

    def launch(x, index, out):
        if (x.shape[1] != D or x.dtype != dtype or index.dtype != index_dtype or
                x.stride(0) != x_stride):
            raise ValueError("gather factory specialization does not match its tensor arguments")
        gather(x, index, out, backend=_backend)

    return launch


def norm_tl(d, dtype, return_dtype, reduce_dtype, has_residual, eps, out_scale,
            do_fp8_cast, num_per_channels, use_tma_aligned_col_major_sf,
            round_sf, use_packed_ue8m0, return_origin_out, xdim, shared_sf_block_m,
            x_stride_s, y_stride_s, residual_stride_s, residual_out_stride_s,
            y_origin_stride_s, sf_stride_s):
    _backend = _selected['norm_tl']
    _norm_backend = _backend
    import torch
    from hcu_adapter.kernels.norm import norm

    if not 0 < d <= 8192 or dtype not in ("float", "bfloat16") or reduce_dtype != "float":
        raise ValueError("Unsupported norm dimension or dtype specialization")
    if xdim not in (2, 3):
        raise ValueError("norm xdim must be 2 or 3")
    if use_packed_ue8m0 and not round_sf:
        raise ValueError("Packed UE8M0 scales require power-of-two rounding")
    if do_fp8_cast:
        if num_per_channels not in (32, 128) or d % num_per_channels:
            raise NotImplementedError("HCU norm supports FP8 groups of 32 or 128 dividing D")
        if str(return_dtype) != "float8_e4m3fn":
            raise ValueError("Quantized norm requires float8_e4m3fn output")
    elif num_per_channels or return_dtype != dtype:
        raise ValueError("Unquantized norm must preserve dtype and have no quantization group")
    if shared_sf_block_m and (not do_fp8_cast or not round_sf or num_per_channels != 32):
        raise ValueError("Shared mega-MoE scales require rounded per-32 FP8")
    expected_dtype = torch.float32 if dtype == "float" else torch.bfloat16

    def launch(x, out, weight, residual, residual_out, sf, extra_sf, origin):
        if x.dtype != expected_dtype or x.shape[-1] != d or (xdim == 2 and x.shape[0] != 1):
            raise ValueError("norm factory specialization does not match its input")
        pairs = [(x, x_stride_s), (out, y_stride_s)]
        if has_residual:
            pairs.extend(((residual, residual_stride_s), (residual_out, residual_out_stride_s)))
        if return_origin_out:
            pairs.append((origin, y_origin_stride_s))
        if do_fp8_cast and use_packed_ue8m0 and not use_tma_aligned_col_major_sf:
            pairs.append((sf, sf_stride_s))
        if any(t.stride(1) != stride for t, stride in pairs):
            raise ValueError("norm factory row strides do not match tensor arguments")
        norm(x, out, weight, residual, residual_out, sf, extra_sf, origin,
             backend=_norm_backend, has_residual=has_residual, eps=eps, out_scale=out_scale,
             do_fp8_cast=do_fp8_cast, num_per_channels=num_per_channels,
             col_packed=use_tma_aligned_col_major_sf and use_packed_ue8m0,
             round_sf=round_sf, use_packed_ue8m0=use_packed_ue8m0,
             return_origin_out=return_origin_out, shared_sf_block_m=shared_sf_block_m)

    return launch


def expand_swa_index_tl(width):
    _backend = _selected['expand_swa_index_tl']
    import torch
    from hcu_adapter.kernels.index_ops import expand_swa_index

    if width <= 0:
        raise ValueError("expand_swa_index width must be positive")

    def launch(pivots, out):
        if (pivots.dtype != torch.int32 or out.dtype != torch.int32 or pivots.shape[1] != 4
                or out.shape != (pivots.shape[0], width)):
            raise ValueError("expand_swa_index factory specialization does not match its tensor arguments")
        expand_swa_index(pivots, out, backend=_backend)

    return launch


def fused_norm_rope_hadamard_cast_store_kvcache_tl(
        num_heads, head_dim, *, do_norm, norm_eps, do_rope, rotary_dim, rope_offset,
        positions_dtype, is_neox, conjugate, do_hadamard, hadamard_scale, cast_fmt,
        num_per_channels, round_sf, sf_format, return_bf16, bf16_stage, return_quant,
        cache_layout, block_size, small_batch):
    _backend = _selected['fused_norm_rope_hadamard_cast_store_kvcache_tl']
    _norm_backend = _backend
    from hcu_adapter.kernels.fused_transform import fused_transform
    if _norm_backend == 'lightop':
        from hcu_adapter.kernels.fused_transform import fused_transform_lightop as fused_transform

    if do_hadamard:
        raise NotImplementedError("HCU fused transform: Hadamard stage is not implemented (no V4.1 caller)")
    if cast_fmt and (sf_format, round_sf) not in (("ue8m0", True), ("e4m3", False),
                                                  ("fp32", True), ("fp32", False)):
        raise ValueError("Unsupported fused transform scale format")
    scale_format = "fp32_rounded" if (sf_format, round_sf) == ("fp32", True) else sf_format
    if cast_fmt == "e2m1" and (num_per_channels, sf_format) not in ((32, "ue8m0"), (16, "e4m3")):
        raise ValueError("FP4 cast requires g32/UE8M0 or g16/E4M3")
    if do_rope and rope_offset + rotary_dim != head_dim:
        raise ValueError("RoPE must cover the trailing rotary_dim columns")
    store_cache = cache_layout is not None
    if store_cache and (num_heads != 1 or not cast_fmt):
        raise ValueError("KV-cache stores require one head and a cast")
    options = dict(backend=_backend, do_norm=do_norm, norm_eps=norm_eps, do_rope=do_rope, rotary_dim=rotary_dim,
                   rope_offset=rope_offset, is_neox=is_neox, conjugate=conjugate,
                   cast_fmt=cast_fmt, num_per_channels=num_per_channels, sf_format=scale_format,
                   return_bf16=return_bf16, bf16_stage=bf16_stage, return_quant=return_quant,
                   store_cache=store_cache, block_size=block_size)

    def launch(x, weight, positions, cos, sin, out_bf16, out_quant, out_sf,
               cache_quant, cache_sf, slot_mapping):
        if x.shape[1:] != (num_heads, head_dim):
            raise ValueError("fused transform factory specialization does not match its input")
        fused_transform(x, weight, positions, cos, sin, out_bf16, out_quant, out_sf,
                        cache_quant, cache_sf, slot_mapping, **options)

    return launch


def load_paged_kv_cache_tl(layout, block_size, fp4_value_block_stride=0, fp4_value_stride=0,
                           fp4_scale_stride=0):
    _backend = _selected['load_paged_kv_cache_tl']
    from hcu_adapter.kernels.kv_cache_emu import load_paged

    is_quantized = layout.quantized_dtype is not None
    if block_size <= 0 or (is_quantized and layout.bf16_elements):
        raise ValueError("Unsupported paged-cache load specialization")
    mode = "cast" if layout.do_cast else ("index" if is_quantized else "bf16")
    is_fp4_cast = mode == "cast" and layout.quantized_dtype == "e2m1"
    if is_fp4_cast != (fp4_value_stride > 0):
        raise ValueError("FP4 cast loads require logical FP4 strides")
    options = dict(block_size=block_size, mode=mode, fmt=layout.quantized_dtype,
                   group=layout.elements_per_sf, sf_format=layout.sf_format,
                   valid_groups=layout.num_scales)

    def launch(load_segments, kv, kv_sf, cache_value, cache_sf):
        load_paged(load_segments, kv, kv_sf, cache_value, cache_sf, **options)

    return launch


def store_kv_cache_tl(layout, block_size, fp4_value_block_stride=0, fp4_value_stride=0,
                      fp4_scale_stride=0):
    _backend = _selected['store_kv_cache_tl']
    from hcu_adapter.kernels.kv_cache_emu import store_paged

    is_fp4 = layout.quantized_dtype == "e2m1"
    if block_size <= 0 or is_fp4 != (fp4_value_stride > 0):
        raise ValueError("Unsupported paged-cache store specialization")
    options = dict(block_size=block_size, fmt=layout.quantized_dtype,
                   group=layout.elements_per_sf, sf_format=layout.sf_format)

    def launch(kv, cache_value, cache_sf, slot_mapping):
        store_paged(kv, cache_value, cache_sf, slot_mapping, **options)

    return launch


def compress_and_update_state_prefill_tl(DIM, COMPRESS_RATIO, HAS_INDEX):
    _backend = _selected['compress_and_update_state_prefill_tl']
    from hcu_adapter.kernels.compress_emu import compress_prefill

    if COMPRESS_RATIO not in (1, 2):
        raise ValueError("compress ratio must be 1 or 2")

    def launch(score, latent, state_cache, state_block_idx, compressed_batch_mapping, cu_seqlens_q,
               cu_seqlens_k, cu_compressed_seqlens_q, mirror_cp_reverse_index, kv_compressed):
        if score.shape[1] != DIM or (mirror_cp_reverse_index is not None) != HAS_INDEX:
            raise ValueError("compress prefill specialization does not match its arguments")
        compress_prefill(score, latent, state_cache, state_block_idx, compressed_batch_mapping, cu_seqlens_q,
                         cu_seqlens_k, cu_compressed_seqlens_q, mirror_cp_reverse_index, kv_compressed,
                         ratio=COMPRESS_RATIO)

    launch.pass_configs = {}
    return launch


def compress_and_update_state_decode_tl(DIM, COMPRESS_RATIO):
    _backend = _selected['compress_and_update_state_decode_tl']
    from hcu_adapter.kernels.compress_emu import compress_decode

    if COMPRESS_RATIO not in (1, 2):
        raise ValueError("compress ratio must be 1 or 2")

    def launch(score, latent, state_cache, state_block_idx, context_lens, kv_compressed):
        if score.shape[1] != DIM:
            raise ValueError("compress decode specialization does not match its arguments")
        compress_decode(score, latent, state_cache, state_block_idx, context_lens, kv_compressed,
                        ratio=COMPRESS_RATIO)

    return launch


# The upstream interface sets JIT pass configs on the factory before calling it.
compress_and_update_state_prefill_tl.pass_configs = {}
compress_and_update_state_decode_tl.pass_configs = {}


def kv_index_cat_tl(dtype, window_size, topk_tokens, out_stride):
    _backend = _selected['kv_index_cat_tl']
    from hcu_adapter.kernels.index_ops_emu import kv_index_cat

    def launch(kv_index, kv_ca_index, out_swa, out_ca, total_k_swa, total_k):
        if kv_index.shape[1] != window_size or kv_ca_index.shape[1] != topk_tokens:
            raise ValueError("kv_index_cat specialization does not match its arguments")
        kv_index_cat(kv_index, kv_ca_index, out_swa, out_ca, total_k_swa, total_k)

    return launch


def convert_kv_index_tl(seqlen_q, topk_tokens, max_blocks, block_size):
    _backend = _selected['convert_kv_index_tl']
    from hcu_adapter.kernels.index_ops_emu import convert_kv_index

    def launch(topk_idxs, context_lens, block_tables):
        if topk_idxs.shape[1:] != (seqlen_q, topk_tokens) or block_tables.shape[1] != max_blocks:
            raise ValueError("convert_kv_index specialization does not match its arguments")
        convert_kv_index(topk_idxs, context_lens, block_tables, block_size)

    return launch


def candidate_block_max_tl(max_model_len, block_size):
    _backend = _selected['candidate_block_max_tl']
    from hcu_adapter.kernels.index_ops_emu import candidate_block_max

    def launch(logits, valid_lens, block_scores, valid_blocks):
        if logits.shape[1] != max_model_len:
            raise ValueError("candidate_block_max specialization does not match its arguments")
        candidate_block_max(logits, valid_lens, block_scores, valid_blocks, block_size=block_size)

    return launch


def get_candidate_compact_lens_tl(topk_blocks, block_size):
    _backend = _selected['get_candidate_compact_lens_tl']
    from hcu_adapter.kernels.index_ops_emu import candidate_compact_lens

    def launch(candidate_blocks, valid_lens, compact_lens):
        candidate_compact_lens(candidate_blocks, valid_lens, compact_lens, block_size=block_size)

    return launch


def restore_candidate_indices_inplace_tl(topk_tokens, topk_blocks, block_size, has_offset):
    _backend = _selected['restore_candidate_indices_inplace_tl']
    from hcu_adapter.kernels.index_ops_emu import restore_candidate_indices

    def launch(compact_indices, candidate_blocks, output_idx_offset):
        restore_candidate_indices(compact_indices, candidate_blocks, output_idx_offset,
                                  block_size=block_size, has_offset=has_offset)

    return launch


def fused_sample_tl(vocab_size, do_sample, do_reject_sampling):
    _backend = _selected['fused_sample_tl']
    from hcu_adapter.kernels.sampling_emu import fused_sample

    if do_reject_sampling and not do_sample:
        raise ValueError("reject sampling requires sampling")

    def launch(logits, temperatures, top_ps, top_ks, sample_positions, seq_seeds, is_verify, logprobs,
               candidate_token_ids, last_nextn_token_logprobs, mtp_info_read_rows, last_next_n_token_ids,
               accept_result_verify, draft_selected_logprobs, accept_probs, seed_offset, lse_workspace):
        if logits.shape[1] != vocab_size:
            raise ValueError("fused_sample specialization does not match its arguments")
        fused_sample(logits, temperatures, top_ps, top_ks, sample_positions, seq_seeds, is_verify, logprobs,
                     candidate_token_ids, last_nextn_token_logprobs, mtp_info_read_rows, last_next_n_token_ids,
                     accept_result_verify, draft_selected_logprobs, accept_probs, seed_offset,
                     do_sample=do_sample, do_reject_sampling=do_reject_sampling)

    return launch


def random_sampling_tl(vocab_size, is_nextn_predict):
    _backend = _selected['random_sampling_tl']
    from hcu_adapter.kernels.sampling_emu import random_sampling

    def launch(probs, seq_seeds, sample_positions, out):
        if probs.shape[1] != vocab_size:
            raise ValueError("random_sampling specialization does not match its arguments")
        out.copy_(random_sampling(probs, seq_seeds, sample_positions, is_nextn_predict))

    return launch


def prepare_v4_decode_metadata_tl(max_seqlen_q, max_num_blocks, window_size, swa_block_size, block_size,
                                  force_load_balance, ca_ratios, dense_ratios, sparse_ratios):
    _backend = _selected['prepare_v4_decode_metadata_tl']
    if _backend == "triton":
        from hcu_adapter.kernels.zos_metadata_triton import prepare_v4_decode_metadata
    else:
        from hcu_adapter.kernels.zos_metadata_emu import prepare_v4_decode_metadata

    options = dict(max_num_blocks=max_num_blocks, window_size=window_size, swa_block_size=swa_block_size,
                   block_size=block_size, force_load_balance=force_load_balance, ca_ratios=tuple(ca_ratios),
                   dense_ratios=tuple(dense_ratios), sparse_ratios=tuple(sparse_ratios))

    def launch(*args):
        prepare_v4_decode_metadata(*args, **options)

    return launch


def copy_block_tables_tl(max_seqlen_q, max_num_blocks, BLOCK_SIZE=1024):
    _backend = _selected['copy_block_tables_tl']
    if _backend == "triton":
        from hcu_adapter.kernels.zos_metadata_triton import copy_block_tables
    else:
        from hcu_adapter.kernels.zos_metadata_emu import copy_block_tables

    def launch(seqlens_q, cu_seqlens_q, block_tables_in, block_tables_offsets, block_tables_out):
        copy_block_tables(seqlens_q, cu_seqlens_q, block_tables_in, block_tables_offsets, block_tables_out,
                          max_num_blocks=max_num_blocks)

    return launch


def gather_input_ids_zos_tl(max_seqlen_q):
    _backend = _selected['gather_input_ids_zos_tl']
    if _backend == "triton":
        from hcu_adapter.kernels.zos_metadata_triton import gather_input_ids
    else:
        from hcu_adapter.kernels.zos_metadata_emu import gather_input_ids

    def launch(fresh_tokens, prev_sampled, prev_predict, remap, seqlens_q, cu_seqlens_q, out_input_ids):
        gather_input_ids(fresh_tokens, prev_sampled, prev_predict, remap, seqlens_q, cu_seqlens_q, out_input_ids,
                         max_seqlen_q=max_seqlen_q)

    return launch


def gather_engram_tokens_zos_tl(max_seqlen_q, max_ngram_size, engram_compressed_pad_token, image_sentinel):
    _backend = _selected['gather_engram_tokens_zos_tl']
    if _backend == "triton":
        from hcu_adapter.kernels.zos_metadata_triton import gather_engram_tokens
    else:
        from hcu_adapter.kernels.zos_metadata_emu import gather_engram_tokens

    def launch(*args):
        gather_engram_tokens(*args, max_seqlen_q=max_seqlen_q, max_ngram_size=max_ngram_size,
                             engram_compressed_pad_token=engram_compressed_pad_token, image_sentinel=image_sentinel)

    return launch


def compute_seqlens_k_zos_tl(nextn):
    _backend = _selected['compute_seqlens_k_zos_tl']
    if _backend == "triton":
        from hcu_adapter.kernels.zos_metadata_triton import compute_seqlens_k
    else:
        from hcu_adapter.kernels.zos_metadata_emu import compute_seqlens_k

    def launch(seqlens_k, seqlens_q, remap, prev_num_accepted_tokens):
        compute_seqlens_k(seqlens_k, seqlens_q, remap, prev_num_accepted_tokens, nextn=nextn)

    return launch


def load_kv_index_tl(index_length, block_size, has_block_tables):
    _backend = _selected['load_kv_index_tl']
    from hcu_adapter.kernels.index_ops_emu import load_kv_index

    def launch(kv_index_offset_len, kv_index, flatten_block_tables):
        if kv_index.shape[1] != index_length:
            raise ValueError("load_kv_index specialization does not match its arguments")
        load_kv_index(kv_index_offset_len, kv_index, flatten_block_tables, block_size=block_size,
                      has_block_tables=has_block_tables)

    return launch


def prepare_verification_metadata_tl(num_steps):
    _backend = _selected['prepare_verification_metadata_tl']
    from hcu_adapter.kernels.verification_emu import prepare_verification_metadata

    del num_steps  # the upstream kernel ignores it too

    def launch(ftp, seqlen_q, row_to_req, local_step, is_verify):
        prepare_verification_metadata(ftp, seqlen_q, row_to_req, local_step, is_verify)

    return launch


def expand_sampling_params_tl():
    _backend = _selected['expand_sampling_params_tl']
    from hcu_adapter.kernels.verification_emu import expand_sampling_params

    return expand_sampling_params


def map_lookup_tl(force_topk, mode="by_key", value_dtype=None):
    _backend = _selected['map_lookup_tl']
    import torch

    from hcu_adapter.kernels.verification_emu import map_lookup

    if mode not in ("by_key", "by_index"):
        raise ValueError("map_lookup mode must be by_key or by_index")
    value_dtype = torch.float32 if value_dtype is None else value_dtype
    if mode == "by_key" and not value_dtype.is_floating_point:
        raise ValueError("by_key requires floating-point values")

    def launch(key, value, query, out):
        if value.shape[1] != force_topk or value.dtype != value_dtype:
            raise ValueError("map_lookup specialization does not match its arguments")
        map_lookup(key, value, query, out, by_key=mode == "by_key")

    return launch


def remap_draft_topk_tl(force_topk, vocab_size):
    _backend = _selected['remap_draft_topk_tl']
    from hcu_adapter.kernels.verification_emu import remap_draft_topk

    def launch(draft_topk_index, draft_topk_logprobs, draft_token_ids_pool, target_topk_index, read_rows,
               aligned_logprobs, draft_local_ids, identity_rows):
        if target_topk_index.shape[1] != force_topk:
            raise ValueError("remap_draft_topk specialization does not match its arguments")
        remap_draft_topk(draft_topk_index, draft_topk_logprobs, draft_token_ids_pool, target_topk_index, read_rows,
                         aligned_logprobs, draft_local_ids, identity_rows)

    return launch


def get_accept_result_tl(has_golden):
    _backend = _selected['get_accept_result_tl']
    from hcu_adapter.kernels.verification_emu import get_accept_result

    def launch(*args):
        get_accept_result(*args, has_golden=has_golden)

    return launch


def finalize_verification_tl(has_sum_smaller_probs, has_confidence, num_steps, num_stop_tokens):
    _backend = _selected['finalize_verification_tl']
    from hcu_adapter.kernels.verification_emu import finalize_verification

    def launch(*args):
        if args[14].shape[1] != num_stop_tokens:
            raise ValueError("finalize_verification specialization does not match its arguments")
        finalize_verification(*args, has_sum_smaller_probs=has_sum_smaller_probs, has_confidence=has_confidence,
                              num_steps=num_steps)

    return launch


def sum_smaller_probs_tl(vocab_size):
    _backend = _selected['sum_smaller_probs_tl']
    from hcu_adapter.kernels.verification_emu import sum_smaller_probs

    def launch(logprobs, sample_results, output):
        if logprobs.shape[1] != vocab_size:
            raise ValueError("sum_smaller_probs specialization does not match its arguments")
        sum_smaller_probs(logprobs, sample_results, output)

    return launch


def confidence_head_fused_tl(n_embed, markov_rank, has_markov_emb):
    _backend = _selected['confidence_head_fused_tl']
    from hcu_adapter.kernels.mtp_emu import confidence_head

    def launch(hidden, markov, weight, out, temperature):
        if hidden.shape[1] != n_embed or (markov is not None) != has_markov_emb:
            raise ValueError("Confidence-head specialization does not match its tensors")
        if has_markov_emb and markov.shape != (hidden.shape[0], markov_rank):
            raise ValueError("Confidence-head Markov shape mismatch")
        confidence_head(hidden, markov, weight, out, temperature)

    return launch


def dflash_input_store_tl(n_embed, hc_mult):
    _backend = _selected['dflash_input_store_tl']
    from hcu_adapter.kernels.mtp_emu import dflash_input_store

    def launch(hidden, out):
        if hidden.shape[1:] != (hc_mult, n_embed):
            raise ValueError("DFlash input-store specialization mismatch")
        dflash_input_store(hidden, out)

    return launch


def copy_dflash_decode_input_tl(has_driver_inputs):
    _backend = _selected['copy_dflash_decode_input_tl']
    from hcu_adapter.kernels.mtp_emu import copy_decode_input

    def launch(*args):
        copy_decode_input(*args, has_driver_inputs=has_driver_inputs)

    return launch


def store_mtp_info_tl(nextn, vocab_size, has_index, has_confidence):
    _backend = _selected['store_mtp_info_tl']
    from hcu_adapter.kernels.mtp_triton import store_proposals

    def launch(slots, tokens, probs, index, confidence, token_pool, prob_pool, index_pool, confidence_pool):
        if tokens.shape[1] != nextn or probs.shape[-1] != vocab_size:
            raise ValueError("MTP proposal-store specialization mismatch")
        if (index is not None) != has_index or (confidence is not None) != has_confidence:
            raise ValueError("MTP proposal-store optional inputs mismatch")
        store_proposals(slots, tokens, probs, index, confidence, token_pool, prob_pool, index_pool, confidence_pool)

    return launch


def compute_seqlens_q_by_confidence_rates_tl(nextn):
    _backend = _selected['compute_seqlens_q_by_confidence_rates_tl']
    from hcu_adapter.kernels.mtp_emu import schedule_by_confidence

    def launch(remap, previous, cumulative, seqlens_q, total_q):
        if previous.shape[1] != nextn or cumulative.shape[1] != nextn:
            raise ValueError("MTP confidence scheduling specialization mismatch")
        schedule_by_confidence(remap, previous, cumulative, seqlens_q, total_q)

    return launch


def get_fused_dflash_get_input_tl(swa_block_size, padded_kv_len, has_valid_token_mask, has_gate_force_random):
    _backend = _selected['get_fused_dflash_get_input_tl']
    from hcu_adapter.kernels.mtp_emu import dflash_get_input

    def launch(*args):
        if args[9].shape[1] != padded_kv_len:
            raise ValueError("DFlash KV index specialization mismatch")
        dflash_get_input(*args, swa_block_size=swa_block_size, has_valid_token_mask=has_valid_token_mask,
                         has_gate_force_random=has_gate_force_random)

    return launch


def select_state_slice_tl(block_size_kvstore, block_size_inference, num_layers, feature_dim):
    _backend = _selected['select_state_slice_tl']
    from hcu_adapter.kernels.state_and_constraints_emu import select_state_slice

    def launch(state, cache, last_hit_position):
        if state.shape != (num_layers, block_size_inference, feature_dim) or cache.shape != (num_layers, block_size_kvstore, feature_dim):
            raise ValueError("State-slice specialization mismatch")
        select_state_slice(state, cache, last_hit_position)

    return launch


def load_state_to_cache_tl(block_size_kvstore, block_size_inference, num_layers, feature_dim, vec=None, threads=256):
    _backend = _selected['load_state_to_cache_tl']
    from hcu_adapter.kernels.state_and_constraints_emu import load_state_to_cache

    def launch(state, cache, last_hit_position):
        if state.shape != (num_layers, block_size_inference, feature_dim) or cache.shape != (num_layers, block_size_kvstore, feature_dim):
            raise ValueError("State-load specialization mismatch")
        load_state_to_cache(state, cache, last_hit_position)

    return launch


def apply_token_bitmask_by_step_inplace_tl(vocab_size, has_ftp):
    _backend = _selected['apply_token_bitmask_by_step_inplace_tl']
    from hcu_adapter.kernels.state_and_constraints_triton import mask_logits

    def launch(logits, bitmask, indices, first, step):
        if logits.shape[1] != vocab_size or (first is not None) != has_ftp:
            raise ValueError("Token-bitmask specialization mismatch")
        mask_logits(logits, bitmask, indices, first, step)

    return launch


def detect_outlier_and_abort_tl(dtype, max_abs):
    _backend = _selected['detect_outlier_and_abort_tl']
    from hcu_adapter.kernels.state_and_constraints_emu import detect_outlier

    def launch(data, identifier):
        if data.dtype != dtype:
            raise ValueError("Outlier-detection specialization dtype mismatch")
        detect_outlier(data, identifier, max_abs)

    return launch


def replace_input_embeds_tl(hidden):
    _backend = _selected['replace_input_embeds_tl']
    from hcu_adapter.kernels.vision_emu import replace_input_embeds

    def launch(*args):
        if args[0].shape[1] != hidden:
            raise ValueError("Image embedding specialization mismatch")
        replace_input_embeds(*args)

    return launch


def rgb_pad_to_patches_tl(patch_size):
    _backend = _selected['rgb_pad_to_patches_tl']
    from hcu_adapter.kernels.vision_triton import rgb_pad_to_patches

    def launch(*args):
        if args[-1].shape[1:] != (3, patch_size, patch_size):
            raise ValueError("RGB patch specialization mismatch")
        rgb_pad_to_patches(*args, patch_size=patch_size)

    return launch


def batch_downsample_tl(downsample_ratio, dim, do_downsample, dtype):
    _backend = _selected['batch_downsample_tl']
    from hcu_adapter.kernels.vision_emu import batch_downsample

    def launch(*args):
        if args[0].shape[1] != dim or args[0].dtype != dtype:
            raise ValueError("Vision downsample specialization mismatch")
        batch_downsample(*args, ratio=downsample_ratio, do_downsample=do_downsample)

    return launch


def prepare_cp_mirror_tl(cp_size, compress_ratios):
    _backend = _selected['prepare_cp_mirror_tl']
    from hcu_adapter.kernels.context_parallel_rccl import prepare_mirror

    def launch(*args):
        prepare_mirror(*args, cp_size=cp_size, ratios=tuple(compress_ratios))

    return launch


def yoco_cp_boundary_fused_tl(cp_size):
    _backend = _selected['yoco_cp_boundary_fused_tl']
    from hcu_adapter.kernels.context_parallel_rccl import yoco_boundary

    def launch(*args):
        yoco_boundary(*args, cp_size=cp_size)

    return launch
