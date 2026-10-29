"""Factory manifest pinned to deepinfer 774ce24; missing kernels fail on call."""

from hcu_adapter.bridges._deepinfer.tilelang_ops import (  # noqa: F401
    expand_swa_index_tl, fused_norm_rope_hadamard_cast_store_kvcache_tl, gather_tl,
    candidate_block_max_tl, compress_and_update_state_decode_tl, compress_and_update_state_prefill_tl,
    convert_kv_index_tl, fused_sample_tl, get_candidate_compact_lens_tl, kv_index_cat_tl, random_sampling_tl,
    restore_candidate_indices_inplace_tl, prepare_v4_decode_metadata_tl, copy_block_tables_tl,
    gather_input_ids_zos_tl, gather_engram_tokens_zos_tl, compute_seqlens_k_zos_tl, load_kv_index_tl,
    load_paged_kv_cache_tl, norm_tl, store_kv_cache_tl, prepare_verification_metadata_tl, expand_sampling_params_tl,
    map_lookup_tl, remap_draft_topk_tl, get_accept_result_tl, finalize_verification_tl, sum_smaller_probs_tl)
from hcu_adapter.bridges._deepinfer.tilelang_ops import (
    confidence_head_fused_tl, dflash_input_store_tl, copy_dflash_decode_input_tl, store_mtp_info_tl, compute_seqlens_q_by_confidence_rates_tl, get_fused_dflash_get_input_tl, select_state_slice_tl, load_state_to_cache_tl, apply_token_bitmask_by_step_inplace_tl, detect_outlier_and_abort_tl
)
from hcu_adapter.bridges._deepinfer.tilelang_ops import (
    batch_downsample_tl, prepare_cp_mirror_tl, replace_input_embeds_tl, rgb_pad_to_patches_tl, yoco_cp_boundary_fused_tl
)
from hcu_adapter.bridges._deepinfer.hcu_config import PLANNED_TILELANG_FACTORIES


def _unsupported(name):
    def factory(*args, **kwargs):
        raise NotImplementedError(
            "HCU tilelang_ops." + name + " is not implemented. "
            "Add a verified provider in kernels/ and register it; see docs/config.md (R-06).")

    factory.__name__ = name
    # compress's interface sets the JIT configuration before calling its factory.
    factory.pass_configs = {}
    return factory


for _name in PLANNED_TILELANG_FACTORIES:
    globals()[_name] = _unsupported(_name)
