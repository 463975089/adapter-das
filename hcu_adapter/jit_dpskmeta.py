# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JIT DSV4-family portability needles — DTK hipcc vs CUDA-isms.

Compiler-input transformation: _apply operates only on private adapter staging
assets via jit_sources. install() delegates to that complete bundle pipeline;
no community source or module origin is rewritten.


Upstream v0.5.22 JIT-compiles the DeepSeek-V4 attention/indexer/MoE kernel
family from `sglang/kernels/jit/csrc/deepseek_v4/`. On DTK gfx938 several
CUDA-isms kill the hipcc build. Failure classes (all with landed precedents
in earlier needles, except where noted):

1. NVIDIA cub include -> DTK hipCUB umbrella + `namespace cub = hipcub`
   alias (hipCUB's AMD path declares the API in namespace hipcub).
2. mask-form shuffles (`__shfl*_sync`) -> non-mask HIP forms; explicit
   width=32 where CUDA 32-lane warp segmentation matters on wave64.
3. bare `__syncwarp()` -> `__builtin_amdgcn_wave_barrier()` under #if
   (DTK has no __syncwarp in any form).
4. structured binds into HIP vector types (float2/uint2, anonymous union)
   -> member reads; the oldest ABI fact on this stack.
5. `%laneid` PTX -> utils.cuh `::sglang::device::get_lane_id()`.
6. `__shared__ alignas(N)` attribute placement -> swapped order.
7. `::cudaMemcpy` -> `::hipMemcpy`.
8. PTX scalar asm (div.full/add/rsqrt.approx.ftz) -> plain ops + rsqrtf.
9. CUDA bit-trick intrinsics `__byte_perm`/`__dp4a` (topk hit packing) and
   bf16 pair masks `__hgt2_mask`/`__heq2_mask` -> semantic-equivalent
   scalar rewrites (same outputs; see needle comments).

Every needle was derived from the live boot/capture chain or the
`dsv4_syntax_check.sh` preflight driver (wrapper-include + hipcc
-fsyntax-only over the whole family), nmz16 2026-10-20. gfx950-only files
(mxfp8_gemv_gfx95 static_asserts by design) and wo_a_fused (Hopper cluster
PTX; wo_a takes the HIP streaming-dequant path instead) are intentionally
NOT patched.

Text-patch discipline per jit_ptgq: needle missing in an existing file
raises loud (upstream changed); file absent skips quietly. Superseded REPLs
are rolled back via _UNDO before re-applying. All writes atomic
(tempfile + os.replace): parent + 8 TP children run install() within
seconds of each other and a torn write cost us a 119-line tail once.
"""

import logging
import os
import pathlib
import tempfile

LOG = logging.getLogger("hcu.adapter")


def _atomic_write(p: pathlib.Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".hcu_dpsk_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# superseded REPLs rolled back to their needle form before re-applying
_UNDO = {
    "deepseek_v4/main_norm_rope.cuh": [
        (
            "  __syncwarp_full_mask();  // HCU portability shim (jit_rope precedent)",
            "  __syncwarp();",
        ),
        (
            "  __syncwarp(0xffffffffu);  // HCU: DTK takes the mask arg",
            "  __syncwarp();",
        ),
    ],
}

_CAST = "cast<fp32x2_t>"

_ALIGNAS_BLOCK = (
    "  __shared__ alignas(16) float seg_kv[kTileDim];\n"
    "  __shared__ alignas(16) float seg_max[kTileDim];\n"
    "  __shared__ alignas(16) float seg_sum[kTileDim];",
    "  alignas(16) __shared__ float seg_kv[kTileDim];   // HCU: attr before storage class\n"
    "  alignas(16) __shared__ float seg_max[kTileDim];\n"
    "  alignas(16) __shared__ float seg_sum[kTileDim];",
)

_PACK_HITS = (
    "  const auto lo = __byte_perm(m[0], m[1], 0x7531);\n"
    "  const auto hi = __byte_perm(m[2], m[3], 0x7531);\n"
    "  const auto nib = __dp4a(static_cast<int>(lo), static_cast<int>(0xF8FCFEFFu), 0);  // -1,-2,-4,-8\n"
    "  return __dp4a(static_cast<int>(hi), static_cast<int>(0x80C0E0F0u), nib);          // -16..-128",
    "  // HCU: CUDA __byte_perm/__dp4a sign-trick -> direct hit-bit packing\n"
    "  // (identical output: bit e set iff element e's flag byte is 0xFF)\n"
    "  uint32_t bits = 0;\n"
    "#pragma unroll\n"
    "  for (uint32_t e = 0; e < 8; ++e)\n"
    "    bits |= (m[e >> 1] & (0xFFu << ((e & 1) * 8))) ? (1u << e) : 0u;\n"
    "  return bits;",
)

_MHC_ASM = (
    '      asm("div.full.f32 %0, %1, %2;" : "=f"(mean) : "f"(sum), "f"(5120.f));\n'
    "      // Match Triton's unqualified PTX add: ptxas may contract the\n"
    "      // constant division's reciprocal multiply with this epsilon add.\n"
    '      asm("add.f32 %0, %1, %2;" : "=f"(mean) : "f"(mean), "f"(p.eps));\n'
    '      asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(mean));',
    "      mean = sum / 5120.f;  // HCU: PTX div.full -> plain rn division\n"
    "      // (epsilon add kept separate from the division, as upstream notes)\n"
    "      mean = mean + p.eps;\n"
    "      inverse = rsqrtf(mean);  // HCU: rsqrt.approx.ftz -> rsqrtf",
)

# Needles whose NVIDIA-only construct may be absent from newer upstream
# trees entirely (upstream self-healed, e.g. sglang main 2026-11 rewrote
# topk_bf16_small.cuh to pass lane_id as a parameter). When the underlying
# construct is verifiably gone from the file the patch has no object and is
# skipped loudly; a changed-but-present construct still trips the §3 gate.
_SELF_HEAL_ABSENT = {
    "deepseek_v4/topk_bf16_small.cuh": {
        '  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));': "%%laneid",
    },
}


_PATCHES = {
    "deepseek_v4/paged_mqa_metadata.cuh": [
        (
            "#include <cub/block/block_scan.cuh>",
            "#include <hipcub/hipcub.hpp>  // HCU: DTK has hipCUB (rocPRIM backend)\n"
            "namespace cub = hipcub;  // HCU: AMD hipCUB mirrors the cub API",
        ),
        (
            "uint32_t y = __shfl_up_sync(0xffffffff, v, o);",
            "uint32_t y = __shfl_up(v, o, 32);  // HCU: no mask-form shuffle, width=32",
        ),
        (
            "running = __shfl_sync(0xffffffff, v, 31);",
            "running = __shfl(v, 31, 32);  // HCU: no mask-form shuffle, width=32",
        ),
    ],
    "deepseek_v4/c_plan.cuh": [
        (
            "    const auto [ragged_id, batch_id] = unpack_w(plan_w);",
            "    const uint2 unpacked_w = unpack_w(plan_w);  // HCU: uint2 not decomposable\n"
            "    const uint32_t ragged_id = unpacked_w.x;\n"
            "    const uint32_t batch_id = unpacked_w.y;",
        ),
    ],
    "deepseek_v4/main_norm_rope.cuh": [
        (
            "  __syncwarp();",
            "#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIP_PLATFORM_AMD__)\n"
            "  __builtin_amdgcn_wave_barrier();  // HCU: wavefront barrier, DTK has no __syncwarp\n"
            "#else\n"
            "  __syncwarp(0xffffffffu);\n"
            "#endif",
        ),
        (
            "  const auto [freq_real, freq_imag] = freq;",
            "  const float freq_real = freq.x;  // HCU: float2 not decomposable\n"
            "  const float freq_imag = freq.y;",
        ),
        # template-body float2 binds (syntax preflight cannot see these;
        # they detonate at kernel instantiation — main_q_norm_rope module,
        # nmz16 2026-10-20)
        (
            "  const auto [x_real, x_imag] = cast<fp32x2_t>(elem);",
            "  const fp32x2_t _re = cast<fp32x2_t>(elem);  // HCU: float2 bind\n"
            "  const auto x_real = _re.x, x_imag = _re.y;",
        ),
        (
            "        const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);",
            "        const float other = __shfl_xor(data[i], mask, kWarpThreads);"
            "  // HCU: no mask-form shuffle",
        ),
    ],
    "deepseek_v4/topk_bf16_small.cuh": [
        (
            '  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));',
            "  // HCU: %laneid is NVIDIA-only; canonical HIP identity below\n"
            "  lane_id = static_cast<uint32_t>(::sglang::device::get_lane_id());",
        ),
        (
            "    const auto t = __shfl_down_sync(device::warp::kFullMask, inc, offset);",
            "    const auto t = __shfl_down(inc, offset);  // HCU: no mask-form shuffle",
        ),
        (
            "  warp_base = __shfl_sync(warp::kFullMask, warp_base, kWarpThreads - 1);",
            "  warp_base = __shfl(warp_base, kWarpThreads - 1);"
            "  // HCU: no mask-form shuffle",
        ),
        _PACK_HITS,
        (
            "      gt[j] = __hgt2_mask(vecs[i][j], pivot_x2);",
            "      {\n"
            "        const float2 _gv = __bfloat1622float2(vecs[i][j]);\n"
            "        const float2 _gp = __bfloat1622float2(pivot_x2);\n"
            "        gt[j] = (_gv.x > _gp.x ? 0xFFFFu : 0u) |"
            " (_gv.y > _gp.y ? 0xFFFF0000u : 0u);  // HCU: __hgt2_mask\n"
            "      }",
        ),
        (
            "      eq[j] = __heq2_mask(vecs[i][j], pivot_x2);",
            "      {\n"
            "        const float2 _ev = __bfloat1622float2(vecs[i][j]);\n"
            "        const float2 _ep = __bfloat1622float2(pivot_x2);\n"
            "        eq[j] = (_ev.x == _ep.x ? 0xFFFFu : 0u) |"
            " (_ev.y == _ep.y ? 0xFFFF0000u : 0u);  // HCU: __heq2_mask\n"
            "      }",
        ),
    ],
    "deepseek_v4/flashmla_sched_meta.cuh": [
        (
            "    partial += __shfl_xor_sync(uint32_t(-1), partial, offset);",
            "    partial += __shfl_xor(partial, offset);  // HCU: no mask-form shuffle",
        ),
    ],
    "deepseek_v4/fused_norm_rope.cuh": [
        (
            "    __syncwarp();",
            "#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIP_PLATFORM_AMD__)\n"
            "    __builtin_amdgcn_wave_barrier();  // HCU: DTK has no __syncwarp\n"
            "#else\n"
            "    __syncwarp(0xffffffffu);\n"
            "#endif",
        ),
        (
            "    const auto [freq_real, freq_imag] = freq;",
            "    const float freq_real = freq.x;  // HCU: float2 not decomposable\n"
            "    const float freq_imag = freq.y;",
        ),
        (
            "    const auto [x_real, x_imag] = cast<fp32x2_t>(elem);",
            "    const fp32x2_t _re = cast<fp32x2_t>(elem);  // HCU: float2 bind\n"
            "    const auto x_real = _re.x, x_imag = _re.y;",
        ),
    ],
    "deepseek_v4/store.cuh": [
        (
            "    const auto [x, y] = cast<fp32x2_t>(elems);",
            "    const fp32x2_t _se = cast<fp32x2_t>(elems);  // HCU\n"
            "    const auto x = _se.x, y = _se.y;",
        ),
        (
            "    const auto [x, y] = cast<fp32x2_t>(elems[i]);",
            "    const fp32x2_t _si = cast<fp32x2_t>(elems[i]);  // HCU\n"
            "    const auto x = _si.x, y = _si.y;",
        ),
        (
            "  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);",
            "  const fp32x2_t _s0 = cast<fp32x2_t>(elems[0]);  // HCU\n"
            "  const auto x0 = _s0.x, x1 = _s0.y;",
        ),
        (
            "  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);",
            "  const fp32x2_t _s1 = cast<fp32x2_t>(elems[1]);  // HCU\n"
            "  const auto y0 = _s1.x, y1 = _s1.y;",
        ),
    ],
    "deepseek_v4/silu_and_mul_masked_post_quant.cuh": [
        (
            "  const auto [g0, g1] = cast<fp32x2_t>(gate);",
            "  const fp32x2_t _sg = cast<fp32x2_t>(gate);  // HCU\n"
            "  const auto g0 = _sg.x, g1 = _sg.y;",
        ),
        (
            "  const auto [u0, u1] = cast<fp32x2_t>(up);",
            "  const fp32x2_t _su = cast<fp32x2_t>(up);  // HCU\n"
            "  const auto u0 = _su.x, u1 = _su.y;",
        ),
        (
            "    const auto [x, y] = silu_and_mul<kApplySwigluLimit>(gate_vec[i], up_vec[i], params.swiglu_limit);",
            "    const fp32x2_t _sm = silu_and_mul<kApplySwigluLimit>(gate_vec[i], up_vec[i], params.swiglu_limit);  // HCU\n"
            "    const auto x = _sm.x, y = _sm.y;",
        ),
    ],
    "deepseek_v4/rope.cuh": [
        (
            "  const auto [f_real, f_imag] = freq_ptr[lane_id];",
            "  const fp32x2_t _fr = freq_ptr[lane_id];  // HCU: float2 not decomposable\n"
            "  const auto f_real = _fr.x, f_imag = _fr.y;",
        ),
        (
            "  const auto [x_real, x_imag] = cast<fp32x2_t>(data);",
            "  const fp32x2_t _xr = cast<fp32x2_t>(data);  // HCU: not decomposable\n"
            "  const auto x_real = _xr.x, x_imag = _xr.y;",
        ),
    ],
    "deepseek_v4/fp4_indexer_rope.cuh": [
        (
            "    const auto [scale, inv_scale] = fp4::block_scale(amax);",
            "    const fp32x2_t _bs = fp4::block_scale(amax);  // HCU: not decomposable\n"
            "    const auto scale = _bs.x, inv_scale = _bs.y;",
        ),
        (
            "    const auto [h0, h1] = cast<fp32x2_t>(head_in[0]);",
            "    const fp32x2_t _hv = cast<fp32x2_t>(head_in[0]);  // HCU\n"
            "    const auto h0 = _hv.x, h1 = _hv.y;",
        ),
        (
            "    const auto [t0, t1] = cast<fp32x2_t>(tail_in[0]);",
            "    const fp32x2_t _tv = cast<fp32x2_t>(tail_in[0]);  // HCU\n"
            "    const auto t0 = _tv.x, t1 = _tv.y;",
        ),
        (
            "    const auto [wh0, wh1] = cast<fp32x2_t>(head_w[0]);",
            "    const fp32x2_t _whv = cast<fp32x2_t>(head_w[0]);  // HCU\n"
            "    const auto wh0 = _whv.x, wh1 = _whv.y;",
        ),
        (
            "    const auto [wt0, wt1] = cast<fp32x2_t>(tail_w[0]);",
            "    const fp32x2_t _wtv = cast<fp32x2_t>(tail_w[0]);  // HCU\n"
            "    const auto wt0 = _wtv.x, wt1 = _wtv.y;",
        ),
    ],
    "deepseek_v4/fp4_indexer_rope_hip.cuh": [
        # the HIP dispatch target carries its own copies of the four binds
        # (same needle strings as the non-_hip file; allowlist keys on the
        # needle, so no new entries).
        (
            "    const auto [h0, h1] = cast<fp32x2_t>(head_in[0]);",
            "    const fp32x2_t _hv = cast<fp32x2_t>(head_in[0]);  // HCU\n"
            "    const auto h0 = _hv.x, h1 = _hv.y;",
        ),
        (
            "    const auto [t0, t1] = cast<fp32x2_t>(tail_in[0]);",
            "    const fp32x2_t _tv = cast<fp32x2_t>(tail_in[0]);  // HCU\n"
            "    const auto t0 = _tv.x, t1 = _tv.y;",
        ),
        (
            "    const auto [wh0, wh1] = cast<fp32x2_t>(head_w[0]);",
            "    const fp32x2_t _whv = cast<fp32x2_t>(head_w[0]);  // HCU\n"
            "    const auto wh0 = _whv.x, wh1 = _whv.y;",
        ),
        (
            "    const auto [wt0, wt1] = cast<fp32x2_t>(tail_w[0]);",
            "    const fp32x2_t _wtv = cast<fp32x2_t>(tail_w[0]);  // HCU\n"
            "    const auto wt0 = _wtv.x, wt1 = _wtv.y;",
        ),
    ],
    "deepseek_v4/c1.cuh": [
        (
            f"      const auto [x, y] = {_CAST}(input[j]);",
            "      const fp32x2_t _vi = cast<fp32x2_t>(input[j]);  // HCU\n"
            "      const auto x = _vi.x, y = _vi.y;",
        ),
        (
            f"      const auto [wx, wy] = {_CAST}(weight[j]);",
            "      const fp32x2_t _vw = cast<fp32x2_t>(weight[j]);  // HCU\n"
            "      const auto wx = _vw.x, wy = _vw.y;",
        ),
        (
            f"    const auto [x, y] = {_CAST}(latent[j]);",
            "    const fp32x2_t _vl = cast<fp32x2_t>(latent[j]);  // HCU\n"
            "    const auto x = _vl.x, y = _vl.y;",
        ),
        (
            f"      const auto [r0, r1] = {_CAST}(rotated);",
            "      const fp32x2_t _vr = cast<fp32x2_t>(rotated);  // HCU\n"
            "      const auto r0 = _vr.x, r1 = _vr.y;",
        ),
        (
            "      const auto [x, y] = deepseek_v4::fp4::fake_quant_compressed_kv_x2({data[i * 2 + 0], data[i * 2 + 1]}, scale);",
            "      const fp32x2_t _fq = deepseek_v4::fp4::fake_quant_compressed_kv_x2({data[i * 2 + 0], data[i * 2 + 1]}, scale);  // HCU\n"
            "      const auto x = _fq.x, y = _fq.y;",
        ),
    ],
    "deepseek_v4/c2.cuh": [
        (
            f"    const auto [x, y] = {_CAST}(cast<bf16x2_t>(packed));",
            "    const fp32x2_t _vc = cast<fp32x2_t>(cast<bf16x2_t>(packed));  // HCU\n"
            "    const auto x = _vc.x, y = _vc.y;",
        ),
        (
            f"    const auto [wx, wy] = {_CAST}(weight[i]);",
            "    const fp32x2_t _vw = cast<fp32x2_t>(weight[i]);  // HCU\n"
            "    const auto wx = _vw.x, wy = _vw.y;",
        ),
        (
            f"    const auto [x, y] = {_CAST}(out[i]);",
            "    const fp32x2_t _vo = cast<fp32x2_t>(out[i]);  // HCU\n"
            "    const auto x = _vo.x, y = _vo.y;",
        ),
        (
            f"      const auto [r0, r1] = {_CAST}(rotated);",
            "      const fp32x2_t _vr = cast<fp32x2_t>(rotated);  // HCU\n"
            "      const auto r0 = _vr.x, r1 = _vr.y;",
        ),
        (
            "      const auto [x, y] = deepseek_v4::fp4::fake_quant_compressed_kv_x2({staged[i * 2 + 0], staged[i * 2 + 1]}, scale);",
            "      const fp32x2_t _fq = deepseek_v4::fp4::fake_quant_compressed_kv_x2({staged[i * 2 + 0], staged[i * 2 + 1]}, scale);  // HCU\n"
            "      const auto x = _fq.x, y = _fq.y;",
        ),
    ],
    "deepseek_v4/mega_moe_pre_dispatch.cuh": [
        (
            f"        const auto [v0, v1] = {_CAST}(in_vec[i]);",
            "        const fp32x2_t _vm = cast<fp32x2_t>(in_vec[i]);  // HCU\n"
            "        const auto v0 = _vm.x, v1 = _vm.y;",
        ),
    ],
    "deepseek_v4/mhc_post_combine_norm_prefill.cuh": [
        (
            "      const float post = __shfl_sync(0xffffffff, coeff, 16 + i);",
            "      const float post = __shfl(coeff, 16 + i, 32);  // HCU: no mask-form shuffle",
        ),
        (
            "      const float pre = __shfl_sync(0xffffffff, coeff, 20 + i);",
            "      const float pre = __shfl(coeff, 20 + i, 32);  // HCU",
        ),
        (
            "      const float c0 = __shfl_sync(0xffffffff, coeff, i);",
            "      const float c0 = __shfl(coeff, i, 32);  // HCU",
        ),
        (
            "      const float c1 = __shfl_sync(0xffffffff, coeff, 4 + i);",
            "      const float c1 = __shfl(coeff, 4 + i, 32);  // HCU",
        ),
        (
            "      const float c2 = __shfl_sync(0xffffffff, coeff, 8 + i);",
            "      const float c2 = __shfl(coeff, 8 + i, 32);  // HCU",
        ),
        (
            "      const float c3 = __shfl_sync(0xffffffff, coeff, 12 + i);",
            "      const float c3 = __shfl(coeff, 12 + i, 32);  // HCU",
        ),
        (
            "      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));",
            "      sum = __fadd_rn(sum, __shfl_xor(sum, offset));  // HCU: no mask-form shuffle",
        ),
        (
            "    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 2));",
            "    sum = __fadd_rn(sum, __shfl_xor(sum, 2));  // HCU",
        ),
        (
            "    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 1));",
            "    sum = __fadd_rn(sum, __shfl_xor(sum, 1));  // HCU",
        ),
        _MHC_ASM,
    ],
    "deepseek_v4/c128_online.cuh": [_ALIGNAS_BLOCK],
    "deepseek_v4/c128_online_v2.cuh": [
        _ALIGNAS_BLOCK,
        (
            "        RuntimeDeviceCheck(::cudaMemcpy(dst, src, bytes, ::cudaMemcpyHostToDevice));",
            "        RuntimeDeviceCheck(::hipMemcpy(dst, src, bytes, ::hipMemcpyHostToDevice));  // HCU",
        ),
    ],
}


def _apply(root, register):
    """Transform adapter-owned staging files; never pass a community root."""
    import hcu_adapter.registry as _reg

    csrc = root / "csrc"
    for relpath, needles in _PATCHES.items():
        p = csrc / relpath
        if not p.exists():
            LOG.info("jit_dpskmeta: %s absent (tree predates the module)", relpath)
            continue
        text = p.read_text(encoding="utf-8")

        # self-heal trees carrying superseded patches
        for old_repl, original in _UNDO.get(relpath, []):
            if old_repl in text:
                text = text.replace(old_repl, original)

        applied = []
        for needle, repl in needles:
            if repl in text:
                applied.append(needle)
                continue
            if needle not in text:
                marker = _SELF_HEAL_ABSENT.get(relpath, {}).get(needle)
                if marker is not None and marker not in text:
                    LOG.info(
                        "jit_dpskmeta: %s self-healed upstream (no %r left);"
                        " needle skipped",
                        relpath,
                        marker,
                    )
                    continue
                raise RuntimeError(
                    f"jit_dpskmeta: needle missing in {relpath}:"
                    f" {needle[:60]!r} -- upstream changed; re-derive"
                    " (governance §3)"
                )
            text = text.replace(needle, repl)
            applied.append(needle)

        _atomic_write(p, text)

        target = f"text:sglang/kernels/jit/csrc/{relpath}"
        for needle in applied:
            register(
                "platform_detect",
                "dtk-hipcc-portability",
                target=target,
                mechanism=_reg.MECH_TEXT,
                detail=needle,
            )
        LOG.warning(
            "HCU adapter: JIT %s portability patch installed (%d needles)",
            relpath,
            len(needles),
        )


def install():
    """Compatibility entry: prepare the complete immutable JIT input bundle."""
    from .jit_sources import install as install_inputs

    return install_inputs()
