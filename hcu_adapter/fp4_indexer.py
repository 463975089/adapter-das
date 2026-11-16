# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""FP4 cache admission before any model/cache allocation. No framework imports."""
from dataclasses import dataclass

from hcu_adapter import hcu_env
from hcu_adapter.kill_switch import op_disabled

OP = 'vllm_fp4_indexer'
VARIABLE = 'HCU_VLLM_FP4_INDEXER_IMPL'


@dataclass(frozen=True)
class Fp4Plan:
    ratios: tuple[int, ...]
    manager_block: int
    page_records: int = 64
    head_dim: int = 128
    heads: int = 32
    record_bytes: int = 68


def selected():
    if op_disabled(OP):
        return False
    mode, _ = hcu_env.resolve(VARIABLE, host='vllm')
    if mode not in ('stock', 'lightop'):
        raise ValueError(f'{VARIABLE} must be stock or lightop')
    return mode == 'lightop'


def applicable(cfg):
    """Whether this deployment actually uses the FP4 indexer cache.

    ``selected()`` only reads the implementation knob, which is on by default
    for every process. Core enables the MXFP4 indexer cache purely from
    ``indexer_kv_dtype`` (``dsa_indexer_uses_fp4`` resolves ``auto`` to ``fp8``),
    so an FP8 deployment must never enter the FP4 branch: it would be rejected
    by the admission below for unrelated properties such as a PD connector, and
    would fail before any model weight is allocated.
    """
    if not selected():
        return False
    attention = getattr(cfg, 'attention_config', None)
    if attention is None:
        return False
    return attention.resolve_indexer_kv_dtype('fp8') == 'mxfp4'


def _graph_capture_allowed(cfg):
    """Admit the capture mode through the adapter-owned graph contract.

    ``graph_capture_allowed`` is a vLLM ``CompilationConfig`` rule (breakable
    PIECEWISE admitted, full cudagraph rejected), not plugin mechanism, so it
    lives in ``hcu_adapter.v41_fp8_indexer``. That module hands back the
    predicate this process admits by: an activated plugin binds a zero-argument
    hook that reads its historical module object, so a patch there still reaches
    this check. The import is lazy so this admission module stays importable
    before the plugin activates and in adapter-only CPU tests.
    """
    from hcu_adapter.v41_fp8_indexer import resolve_graph_capture_allowed
    return bool(resolve_graph_capture_allowed()(cfg))


def validate_config(cfg):
    """Reject unsupported deployments before native constructors allocate."""
    if not selected():
        return None
    hf = cfg.model_config.hf_text_config
    p = cfg.parallel_config
    ratios = tuple(hf.compress_ratios)
    errors = []
    # Core's V4.1 quant_config accepts the flat and text-config identifiers;
    # fused_moe/config.py pins this same exact V4.1 pair. No prefix matching.
    if getattr(hf, 'model_type', '') not in ('deepseek_v41', 'deepseek_v41_text'):
        errors.append('DeepSeek-V4.1 required')
    if (hf.index_head_dim, hf.index_n_heads, hf.qk_rope_head_dim) != (128, 32, 64):
        errors.append('H32/D128/RoPE64 required')
    if not ratios or any(type(r) is not int or r not in (0, 1, 2) for r in ratios):
        errors.append('compress_ratios must contain only 0/1/2')
    active = set(ratios) - {0}
    if not getattr(hf, 'kv_source_layer_ids', None) or not getattr(
            hf, 'index_source_layer_ids', None):
        errors.append('kv_source_layer_ids and index_source_layer_ids required')
    block = cfg.cache_config.block_size
    if not active:
        errors.append('at least one compressed indexer ratio is required')
    # The allocator owns ``block_size // ratio`` records per positive-ratio
    # page. The decode reader stages supported 32/128-record pages into the
    # fixed 64-record LightOp ABI later.
    if active and any(block % ratio or block // ratio not in (32, 64, 128)
                      for ratio in active):
        errors.append('allocator pages must contain 32/64/128 FP4 records')
    for name in ('tensor_parallel_size', 'pipeline_parallel_size',
                 'prefill_context_parallel_size', 'decode_context_parallel_size'):
        if getattr(p, name) != 1:
            errors.append(name + '=1 required')
    if p.data_parallel_size not in (1, 8):
        errors.append('DP1 or DP8 required')
    if p.enable_dbo or p.enable_eplb:
        errors.append('DBO/EPLB unsupported')
    # LoRA stays rejected. PD/KV-transfer is NOT gated here any more: the FP4
    # page format is carried by the connector as opaque block bytes, so a
    # prefill/decode pair configured identically (both mxfp4) is expected to
    # transfer unchanged. That combination is still unmeasured; it is admitted
    # on request and must be validated before it is treated as supported.
    if cfg.lora_config is not None:
        errors.append('LoRA unsupported')
    if not cfg.use_v2_model_runner or not (
            cfg.model_config.enforce_eager or _graph_capture_allowed(cfg)):
        errors.append('eager MRV2 or PIECEWISE breakable Graph required; '
                      'full cudagraph is not admitted')
    # JIT warmup is NOT rejected here: `register_warmup()` only queues a compile
    # key with `JitWarmupRegistry`, so the owner kernels are still unhashed when
    # the packer rebinds them during model construction. The plugin's
    # `_require_cold()` remains the runtime judge and reports a real conflict.
    if cfg.kernel_config.monokernel:
        errors.append('monokernel unsupported by cold rebinding')
    if cfg.attention_config.indexer_sparse_logits:
        errors.append('dense logits with official candidate masking required')
    if cfg.attention_config.indexer_kv_dtype not in ('auto', 'fp8', 'mxfp4'):
        errors.append('unsupported indexer dtype')
    spec = cfg.speculative_config
    if spec is not None and (spec.method != 'dspark' or spec.num_speculative_tokens != 5
                             or spec.enable_adaptive_verification):
        errors.append('only nonadaptive DSpark(5) or no speculation supported')
    if hcu_env.resolve('HCU_VLLM_SPARSE_INDEXER_IMPL', host='vllm')[0] != 'stock':
        errors.append('C4 Adapter OOT and V41 FP4 cannot own SparseAttnIndexer together')
    if (op_disabled('vllm_fp4_packer')
            or hcu_env.resolve('HCU_VLLM_FP4_PACKER_IMPL', host='vllm')[0] != 'triton'):
        errors.append('explicit triton FP4 lowering required')
    if errors:
        raise ValueError('HCU FP4 admission before allocation: ' + '; '.join(errors))
    return Fp4Plan(tuple(sorted(active)), block)
