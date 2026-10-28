"""slimquant_w4a8 checkpoint support (GLM-5-W4A8_V2_6, GLM-5.2-Channel-INT4-w4a8).

Checkpoint format, reverse-engineered from weight headers under /public/share
and validated against the same-model FP8 checkpoint (DeepSeek-V3-0324, per-row
correlation 1.000 / 0.993):

- routed experts (``.mlp.experts.<i>.{gate,up,down}_proj``): signed int4,
  two nibbles per I8 byte, HIGH nibble on the even k column, stored [N, K//2].
  lmslim's kernels accumulate ``nibble << 4`` int8 payloads and then multiply
  by ``b_scale``, so the stored scale is the true scale / 16.
- every other tensor with a ``.weight_scale`` sibling (attn projections,
  indexer wk/wq_b, dense + shared-expert MLPs): plain W8 int8 [N, K],
  ``true_w = w8 * weight_scale``.
- norms / router / embed / lm_head: bf16, untouched.

Serving strategy (HCU, 64 GB cards cannot hold unpacked experts):

- routed experts stay packed int4 in HBM; a triton kernel ported from lmslim
  (fused_moe_kernel_int4_w4a8) dequantizes in-register and does an int8
  ``tl.dot`` with per-token int8 activations — w4a8 where it counts, ~181 GB
  total instead of ~363 GB;
- other quantized linears are dequantized to bf16 at load (shim below) and
  served by UnquantizedLinearMethod (sgl_kernel's int8_scaled_mm has no DCU
  build; the +3.2 GB/GPU for 6144-wide projections is acceptable);
- ``safetensors.safe_open`` is shimmed inside sglang's weight loader only, so
  no sglang module is modified.

Gated by env ``HCU_SLIMQUANT=1``.

Governance note (2026-10-13): slimquant_w4a8 is an lmslim-private checkpoint
format — upstream sglang (AMD/CUDA) has no such quant_method, so per the
governance boundary test ("does upstream have this behavior?") this is NOT
adapter debt; it lives here as **plugin-track legacy** (predates the dual-repo
rule) and stays only because GLM-5-W4A8 e2e depends on it today. Planned move:
port to the plugin repo (governance §5.1 trajectory); before the move,
document the format contract above (nibble order, scale/16 semantics) as the
plugin's first-class contract tests. Do notextend with new formats here —
new private-format support starts in the plugin repo directly.
"""

import logging
import re
import types

import torch

logger = logging.getLogger(__name__)

_ROUTED_EXPERT_RE = re.compile(
    r"\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$"
)


class _SlimquantShard:
    """Proxies one safetensors file handle, transforming slimquant tensors.

    The transform is gated per checkpoint: slimquant-format models only.
    Other quantized formats that also use ``.weight_scale`` siblings (e.g.
    compressed-tensors int-quantized) must pass through untouched — their
    schemes expect the raw int8 weight + scale, and dequantizing here turns
    every quantized projection into zeros (bf16 copy_ into int8 param).
    """

    _format_cache: dict = {}

    def __init__(self, f, path=""):
        self._f = f
        self._active = _SlimquantShard._checkpoint_is_slimquant(path)
        if self._active:
            self._keys = [k for k in f.keys() if self._keep(k)]
        else:
            self._keys = list(f.keys())

    @classmethod
    def _checkpoint_is_slimquant(cls, path):
        """Read the owning checkpoint dir's config.json quant fields.

        slimquant exports carry quantization_config.quant_method (GLM-5
        family) and/or .format == 'slimquant_w4a8'. Anything else — no
        config, probe error, other quant formats — is passthrough: the
        transform is only ever safe on genuine slimquant checkpoints, and
        misfiring it on a CT style checkpoint loads all-zero int8 weights.
        """
        if not path:
            return False
        cached = cls._format_cache.get(path)
        if cached is not None:
            return cached
        try:
            import json
            import os

            d = os.path.dirname(os.path.abspath(path))
            cfg = None
            for _ in range(3):  # nested-name checkpoints keep config at root
                p = os.path.join(d, "config.json")
                if os.path.exists(p):
                    cfg = json.load(open(p, encoding="utf-8"))
                    break
                d = os.path.dirname(d)
            q = (cfg or {}).get("quantization_config") or {}
            result = (
                str(q.get("quant_method", "")) == "slimquant_w4a8"
                or str(q.get("format", "")) == "slimquant_w4a8"
            )
        except Exception:
            result = False
        cls._format_cache[path] = result
        return result

    @staticmethod
    def _keep(name):
        # Non-expert weights are dequantized to bf16 below, so their scales
        # must not reach the loader (UnquantizedLinearMethod registers no
        # weight_scale param, and the stacked-name rewrite would KeyError).
        if name.endswith("_scale"):
            base = name[: -len("_scale")]
            return bool(_ROUTED_EXPERT_RE.search(base))
        return True

    def keys(self):
        return self._keys

    def metadata(self):
        try:
            return self._f.metadata()
        except Exception:
            return None

    def get_tensor(self, name):
        if not self._active:
            return self._f.get_tensor(name)
        f = self._f
        if name.endswith("_scale"):
            base = name[: -len("_scale")]
            if _ROUTED_EXPERT_RE.search(base):
                # kernel convention: stored scale is true scale / 16
                return f.get_tensor(name) * 16.0
            raise KeyError(name)

        if name.endswith(".weight"):
            if _ROUTED_EXPERT_RE.search(name):
                # keep the int4 packing; the MoE kernel unpacks in-register
                return f.get_tensor(name)
            if name + "_scale" in self._f.keys():
                w = f.get_tensor(name).to(torch.float32)
                s = f.get_tensor(name + "_scale").to(torch.float32)
                return (w * s).to(torch.bfloat16)

        return f.get_tensor(name)

    def __getattr__(self, name):
        return getattr(self._f, name)

    def __enter__(self):
        self._f.__enter__()
        return self

    def __exit__(self, *exc):
        return self._f.__exit__(*exc)


def _wrap_safe_open(real_open):
    def safe_open(*args, **kwargs):
        path = args[0] if args else kwargs.get("filename")
        shard = _SlimquantShard.__new__(_SlimquantShard)
        shard._f = real_open(*args, **kwargs)
        shard._active = _SlimquantShard._checkpoint_is_slimquant(path or "")
        if shard._active:
            shard._keys = [k for k in shard._f.keys() if _SlimquantShard._keep(k)]
        else:
            shard._keys = list(shard._f.keys())
        return shard

    return safe_open


def install_weight_loader_shim():
    """Swap sglang's weight_utils `safetensors` module binding for a shim.

    weight_utils accesses ``safetensors.safe_open`` through the module
    object, so replacing the module reference in that single namespace
    confines the patch to checkpoint iteration.
    """
    import sglang.srt.model_loader.weight_utils as wu

    sv = wu.safetensors
    if getattr(sv, "_hcu_slimquant", False):
        return
    shim = types.ModuleType("safetensors_hcu_slimquant")
    shim.__getattr__ = lambda name: getattr(sv, name)
    shim.safe_open = _wrap_safe_open(sv.safe_open)
    shim._hcu_slimquant = True
    wu.safetensors = shim
    logger.info("[hcu-slimquant] safetensors shim installed in weight_utils")


# ---------------------------------------------------------------------------
# w4a8 fused MoE: triton kernel ported from lmslim's
# fused_moe_kernel_int4_w4a8 (B packed int4 [E, N, K//2], high nibble on even
# k; A per-token int8; C = (A(q) x nibbles(B)) * a_scale * b_scale).
# ---------------------------------------------------------------------------


def _build_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _w4a8_moe_kernel(
        a_ptr,
        b_ptr,  # int8 [E, N, K//2], packed signed int4
        c_ptr,
        a_scale_ptr,  # fp32 [num_tokens*top_k, K//A_GROUP] per-k-group
        b_scale_ptr,  # fp32 [E, N]
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N,
        K,  # unpacked K
        EM,
        num_valid_tokens,
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
        token_mask = offs_token < num_valid_tokens

        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if off_experts == -1:
            return

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
        offs_ak = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_ak[
            None, :
        ] * stride_ak
        offs_bk = tl.arange(0, BLOCK_SIZE_K // 2).to(tl.int64)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bn[:, None] * stride_bn
            + offs_bk[None, :] * stride_bk
        )
        a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k_iter in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b_raw = tl.load(b_ptrs)
            b_u = b_raw.to(tl.uint8)
            hi = (b_u >> 4).to(tl.int8)
            lo = (b_u & 0x0F).to(tl.int8)
            hi = tl.where(hi >= 8, hi - 16, hi)
            lo = tl.where(lo >= 8, lo - 16, lo)
            b4 = tl.trans(tl.interleave(hi, lo))  # [BK, BN], hi nibble on even k
            # per-k-group activation scale applied inside the K loop
            a_scale_k = tl.load(
                a_scale_ptrs + k_iter * stride_ask, mask=token_mask, other=0.0
            )
            accumulator += tl.dot(a, b4) * a_scale_k[:, None]

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

        b_scale = tl.load(b_scale_ptrs)
        accumulator = accumulator * b_scale[None, :]
        if MUL_ROUTED_WEIGHT:
            w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
            accumulator = accumulator * w[:, None]

        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator.to(compute_type), mask=c_mask)

    return _w4a8_moe_kernel


_W4A8_KERNEL = None


def _get_kernel():
    global _W4A8_KERNEL
    if _W4A8_KERNEL is None:
        _W4A8_KERNEL = _build_kernel()
    return _W4A8_KERNEL


def _per_token_group_quant_int8(x, group=64):
    """Dynamic per-token-k-group int8 quant (NPU w4a8 style, no static
    input_scale in these checkpoints). CUDA-graph-safe pure torch ops."""
    M, K = x.shape
    xg = x.float().view(M, K // group, group)
    s = (xg.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-10)
    q = torch.round(xg / s).clamp(-127, 127).to(torch.int8)
    return q.view(M, K), s.view(M, K // group)


def _invoke_w4a8(
    A,
    Bq,
    C,
    A_scale,
    B_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    top_k,
    config,
    mul_routed_weight,
    compute_type,
):
    import triton

    kernel = _get_kernel()
    N, K_half = Bq.shape[1], Bq.shape[2]
    K = K_half * 2
    EM = sorted_token_ids.shape[0]
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel[grid](
        A,
        Bq,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        topk_weights.numel(),
        A.stride(0),
        A.stride(1),
        Bq.stride(0),
        Bq.stride(2),
        Bq.stride(1),
        C.stride(-2),
        C.stride(-1),
        A_scale.stride(0),
        A_scale.stride(1),
        B_scale.stride(0),
        B_scale.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config.get("GROUP_SIZE_M", 8),
        num_warps=config.get("num_warps", 4),
        num_stages=config.get("num_stages", 2),
    )


def fused_experts_w4a8(
    hidden_states,
    w13_q,  # int8 [E, 2I, H//2]
    w2_q,  # int8 [E, H, I//2]
    w13_scale,  # fp32 [E, 2I] (already x16)
    w2_scale,  # fp32 [E, H]
    topk_weights,
    topk_ids,
    routed_scaling_factor=None,
    inplace=False,
):
    import torch.nn.functional as F
    import triton

    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    num_tokens, H = hidden_states.shape
    E, N2, _ = w13_q.shape
    I = N2 // 2
    top_k = topk_ids.shape[1]
    device = hidden_states.device
    compute_type = (
        triton.language.bfloat16
        if hidden_states.dtype == torch.bfloat16
        else triton.language.float16
    )

    if num_tokens <= 16:
        bm = 16
    elif num_tokens <= 64:
        bm = 32
    else:
        bm = 64
    config = {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )

    # up GEMM: per-token-group int8 activation x packed weights -> [tokens, topk, 2I]
    a1_q, a1_scale = _per_token_group_quant_int8(hidden_states)
    cache1 = torch.empty(
        (num_tokens, top_k, N2), device=device, dtype=hidden_states.dtype
    )
    _invoke_w4a8(
        a1_q,
        w13_q,
        cache1,
        a1_scale,
        w13_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k=top_k,
        config=config,
        mul_routed_weight=False,
        compute_type=compute_type,
    )

    # activation: silu(gate) * up. The stacked loader writes w13 as chunk
    # [gate; up] rows (w1 offset 0 / w3 offset I); sglang's plain-silu triton
    # runner is chunk-semantic too (silu_and_mul(cache.view(-1, N))) --
    # gate_up_interleaved is only consumed by the gpt-oss alpha swiglu branch.
    c1 = cache1.view(num_tokens * top_k, N2)
    gate, up = c1[:, :I], c1[:, I:]
    inter = torch.nn.functional.silu(gate) * up  # [tokens*topk, I] bf16

    # down GEMM with routed weight multiply
    a2_q, a2_scale = _per_token_group_quant_int8(inter)
    cache2 = torch.empty(
        (num_tokens, top_k, H), device=device, dtype=hidden_states.dtype
    )
    _invoke_w4a8(
        a2_q,
        w2_q,
        cache2,
        a2_scale,
        w2_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k=1,
        config=config,
        mul_routed_weight=True,
        compute_type=compute_type,
    )

    out = cache2.to(torch.float32).sum(dim=1)
    if routed_scaling_factor not in (None, 1.0):
        out = out * routed_scaling_factor
    out = out.to(hidden_states.dtype)
    if inplace:
        hidden_states.copy_(out)
        out = hidden_states
    return out


# ---------------------------------------------------------------------------
# Quant method wiring
# ---------------------------------------------------------------------------


def register_quant_config():
    from sglang.srt.layers.quantization import QUANTIZATION_METHODS
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.layers.quantization.w8a8_int8 import (
        W8A8Int8Config,
        W8A8Int8MoEMethod,
    )

    class HcuW4A8MoEMethod(W8A8Int8MoEMethod):
        """Experts stay packed int4 [N, K//2]; GEMMs run fused_experts_w4a8."""

        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        ):
            from sglang.srt.layers.moe.fused_moe_triton import (
                FusedMoeWeightScaleSupported,
            )
            from sglang.srt.utils import set_weight_attrs

            w13 = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // 2,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
            set_weight_attrs(w13, extra_weight_attrs)
            layer.register_parameter("w13_weight", w13)

            w2 = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // 2,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
            set_weight_attrs(w2, extra_weight_attrs)
            layer.register_parameter("w2_weight", w2)

            quant_attrs = {
                **extra_weight_attrs,
                "quant_method": FusedMoeWeightScaleSupported.CHANNEL.value,
            }
            w13_scale = torch.nn.Parameter(
                torch.empty(
                    num_experts, 2 * intermediate_size_per_partition, 1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            set_weight_attrs(w13_scale, quant_attrs)
            layer.register_parameter("w13_weight_scale", w13_scale)

            w2_scale = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
                requires_grad=False,
            )
            set_weight_attrs(w2_scale, quant_attrs)
            layer.register_parameter("w2_weight_scale", w2_scale)

            layer.register_parameter("w13_input_scale", None)
            layer.register_parameter("w2_input_scale", None)

        def apply(self, layer, dispatch_output):
            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

            x = dispatch_output.hidden_states
            topk_output = dispatch_output.topk_output
            # routed_scaling_factor is NOT applied here: on HIP with
            # SGLANG_USE_AITER unset, DeepseekV2MoE multiplies it outside
            # (dsv2: "not _is_cuda and not _is_musa and not _use_aiter" ->
            # final *= routed_scaling_factor), so an in-kernel multiply would
            # double it (6.25x = 2.5^2 per layer).
            out = fused_experts_w4a8(
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                topk_output.topk_weights,
                topk_output.topk_ids,
            )
            return StandardCombineInput(hidden_states=out)

    class HcuSlimquantW4A8Config(W8A8Int8Config):
        @classmethod
        def get_name(cls):
            return "slimquant_w4a8"

        @classmethod
        def get_config_filenames(cls):
            return []

        def can_fuse_shared_expert(self):
            # shared experts are dequantized bf16 while routed experts stay
            # packed int4 — they cannot share the int4 fused-MoE kernel.
            return False

        def get_quant_method(self, layer, prefix):
            from sglang.srt.layers.linear import LinearBase
            from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

            if isinstance(layer, FusedMoE):
                return HcuW4A8MoEMethod(self)
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return None

    QUANTIZATION_METHODS["slimquant_w4a8"] = HcuSlimquantW4A8Config
    logger.info("[hcu-slimquant] quant method slimquant_w4a8 registered (w4a8 triton)")

    # _verify_quantization hard-rejects methods outside rocm_supported_quantization
    # on HIP; neutralize that single gate while it runs.
    import sglang.srt.configs.model_config as mc

    orig_verify = mc.ModelConfig._verify_quantization
    orig_is_hip = mc.is_hip

    def _verify_allow(self):
        mc.is_hip = lambda: False
        try:
            return orig_verify(self)
        finally:
            mc.is_hip = orig_is_hip

    mc.ModelConfig._verify_quantization = _verify_allow
    logger.info("[hcu-slimquant] ModelConfig ROCm quant gate neutralized")


def activate():
    install_weight_loader_shim()
    register_quant_config()
