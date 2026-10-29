"""DeepInfer upstream compute interface providers; installed by the host bridge."""

BACKEND_CATALOG = {'deep_gemm.cublaslt_gemm_nt': {'native': {'status': 'candidate',
                                           'implementation': 'hcu_adapter.kernels.gemm.cublaslt_gemm_nt',
                                           'limitations': 'BF16 contiguous aligned-K uses qualified LightOp '
                                                          'provider, FP32/strided fallback uses FP32 torch '
                                                          'matmul; C/D alias and empty-K semantics; FP16/FP8 '
                                                          'operands unsupported; vision model pending'}},
 'tile_kernels.quant.swiglu_forward': {'torch': {'status': 'candidate',
                                                 'implementation': 'hcu_adapter.kernels.quant_emu.swiglu_forward',
                                                 'limitations': 'Standalone BF16/FP32 V4.1 ViT activation; '
                                                                'same FP32/clamp/weight math as fused cast '
                                                                'provider; clamp counting unsupported; model '
                                                                'qualification pending'},
                                       'lightop': {'status': 'candidate',
                                                   'implementation': 'hcu_adapter.kernels.swiglu_lightop.swiglu_forward',
                                                   'limitations': 'Explicit LightOp b2f8c9d; preserve '
                                                                  'residual/clamp/FP32 intermediate and SF '
                                                                  'ABI; quant native only rounded '
                                                                  'block32/128 FP8, other formats counted as '
                                                                  'compatibility; mega-MoE shared SF '
                                                                  'unsupported; finite and model '
                                                                  'qualification recorded separately'}},
 'tile_kernels.utils.dtype_to_str': {'native': {'status': 'candidate',
                                                'implementation': 'hcu_adapter.kernels.runtime_config.dtype_to_str',
                                                'limitations': 'Pinned host map: FP32/BF16/E4M3/int8 '
                                                               'packed-E2M1 only'}},
 'deep_select.topk': {'torch': {'status': 'candidate',
                                'implementation': 'hcu_adapter.kernels.deepselect_emu.topk',
                                'limitations': 'Pinned GPU top-k, native stable sort; no e2e/throughput '
                                               'qualification'},
                      'lightop': {'status': 'candidate',
                                  'implementation': 'hcu_adapter.kernels.deepselect_lightop.topk',
                                  'limitations': 'Explicit LightOp b2f8c9d; preserve residual/clamp/FP32 '
                                                 'intermediate and SF ABI; quant native only rounded '
                                                 'block32/128 FP8, other formats counted as compatibility; '
                                                 'mega-MoE shared SF unsupported; finite and model '
                                                 'qualification recorded separately'}},
 'tilelang_ops.gather_tl': {'triton': {'status': 'candidate',
                                       'implementation': 'hcu_adapter.kernels.gather_triton.gather',
                                       'limitations': 'Pinned int8 factory contract; no model/e2e '
                                                      'qualification'},
                            'torch': {'status': 'candidate',
                                      'implementation': 'hcu_adapter.kernels.gather_emu.gather',
                                      'limitations': 'Pinned int8 factory contract; no model/e2e '
                                                     'qualification'}},
 'tilelang_ops.norm_tl': {'triton': {'status': 'candidate',
                                     'implementation': 'hcu_adapter.kernels.norm_triton.norm',
                                     'limitations': 'BF16/FP32, FP8 per32/per128; no model/e2e '
                                                    'qualification'},
                          'torch': {'status': 'candidate',
                                    'implementation': 'hcu_adapter.kernels.norm_emu.norm',
                                    'limitations': 'BF16/FP32, FP8 per32/per128; no model/e2e qualification'},
                          'lightop': {'status': 'candidate',
                                      'implementation': 'hcu_adapter.kernels.norm_lightop.norm',
                                      'limitations': 'Explicit LightOp b2f8c9d; preserve residual/clamp/FP32 '
                                                     'intermediate and SF ABI; quant native only rounded '
                                                     'block32/128 FP8, other formats counted as '
                                                     'compatibility; mega-MoE shared SF unsupported; finite '
                                                     'and model qualification recorded separately'}},
 'tilelang_ops.expand_swa_index_tl': {'triton': {'status': 'candidate',
                                                 'implementation': 'hcu_adapter.kernels.index_ops.expand_swa_index',
                                                 'limitations': 'Exact int32 contract incl. '
                                                                'wraparound/clipping; no model/e2e '
                                                                'qualification'},
                                      'torch': {'status': 'candidate',
                                                'implementation': 'hcu_adapter.kernels.index_ops._torch_expand_swa_index',
                                                'limitations': 'Exact int32 contract incl. '
                                                               'wraparound/clipping; no model/e2e '
                                                               'qualification'}},
 'tilelang_ops.fused_norm_rope_hadamard_cast_store_kvcache_tl': {'triton': {'status': 'candidate',
                                                                            'implementation': 'hcu_adapter.kernels.fused_transform.fused_transform',
                                                                            'limitations': 'Norm via norm_tl '
                                                                                           'provider, torch '
                                                                                           'RoPE/cast '
                                                                                           'mirroring the '
                                                                                           'pinned CUDA '
                                                                                           'rounding; '
                                                                                           'E4M3/E2M1 with '
                                                                                           'UE8M0/E4M3/FP32 '
                                                                                           'scales; no '
                                                                                           'Hadamard (no '
                                                                                           'V4.1 caller); no '
                                                                                           'model/performance '
                                                                                           'qualification'},
                                                                 'torch': {'status': 'candidate',
                                                                           'implementation': 'hcu_adapter.kernels.fused_transform.fused_transform',
                                                                           'limitations': 'Norm via norm_tl '
                                                                                          'provider, torch '
                                                                                          'RoPE/cast '
                                                                                          'mirroring the '
                                                                                          'pinned CUDA '
                                                                                          'rounding; '
                                                                                          'E4M3/E2M1 with '
                                                                                          'UE8M0/E4M3/FP32 '
                                                                                          'scales; no '
                                                                                          'Hadamard (no V4.1 '
                                                                                          'caller); no '
                                                                                          'model/performance '
                                                                                          'qualification'},
                                                                 'lightop': {'status': 'candidate',
                                                                             'implementation': 'hcu_adapter.kernels.fused_transform.fused_transform_lightop',
                                                                             'limitations': 'Native '
                                                                                            'norm/RoPE and '
                                                                                            'block32/128 '
                                                                                            'FP8; original '
                                                                                            'BF16 stage '
                                                                                            'boundaries, FP4 '
                                                                                            'quantization '
                                                                                            'and V4.1 paged '
                                                                                            'scatter '
                                                                                            'retained; '
                                                                                            'finite and '
                                                                                            'model '
                                                                                            'qualification '
                                                                                            'recorded '
                                                                                            'separately'}},
 'tile_kernels.transform.apply_rotary': {'triton': {'status': 'candidate',
                                                    'implementation': 'hcu_adapter.kernels.rope_triton.apply_rotary',
                                                    'limitations': 'In-place FP32 fma rotation, '
                                                                   'NeoX/interleaved/conjugate; 4D input '
                                                                   'needs mergeable batch/seq strides; no '
                                                                   'model/performance qualification'},
                                         'lightop': {'status': 'candidate',
                                                     'implementation': 'hcu_adapter.kernels.rope_lightop.apply_rotary',
                                                     'limitations': 'Explicit LightOp b2f8c9d; preserve '
                                                                    'residual/clamp/FP32 intermediate and SF '
                                                                    'ABI; quant native only rounded '
                                                                    'block32/128 FP8, other formats counted '
                                                                    'as compatibility; mega-MoE shared SF '
                                                                    'unsupported; finite and model '
                                                                    'qualification recorded separately'}},
 'tile_kernels.quant.per_token_cast_and_cast_back': {'torch': {'status': 'candidate',
                                                               'implementation': 'hcu_adapter.kernels.quant_emu.per_token_cast_and_cast_back',
                                                               'limitations': 'In-place E4M3/E2M1 round trip '
                                                                              'with FP32 or rounded scales; '
                                                                              'no E4M3 scales or model '
                                                                              'qualification'}},
 'tilelang_ops.load_paged_kv_cache_tl': {'triton': {'status': 'candidate',
                                                    'implementation': 'hcu_adapter.kernels.fused_transform.load_paged',
                                                    'limitations': 'Torch gather/scatter of compact '
                                                                   'segments; cast layouts dequantize as '
                                                                   'bf16(float(q)*sf); no model/performance '
                                                                   'qualification'},
                                         'torch': {'status': 'candidate',
                                                   'implementation': 'hcu_adapter.kernels.fused_transform.load_paged',
                                                   'limitations': 'Torch gather/scatter of compact segments; '
                                                                  'cast layouts dequantize as '
                                                                  'bf16(float(q)*sf); no model/performance '
                                                                  'qualification'}},
 'tilelang_ops.store_kv_cache_tl': {'triton': {'status': 'candidate',
                                               'implementation': 'hcu_adapter.kernels.fused_transform.store_paged',
                                               'limitations': 'Torch quantize+scatter mirroring the pinned '
                                                              'CUDA store rounding (division-based UE8M0); '
                                                              'no model/performance qualification'},
                                    'torch': {'status': 'candidate',
                                              'implementation': 'hcu_adapter.kernels.fused_transform.store_paged',
                                              'limitations': 'Torch quantize+scatter mirroring the pinned '
                                                             'CUDA store rounding (division-based UE8M0); no '
                                                             'model/performance qualification'}},
 'flash_mla.fused_norm_rope_attn_rope_cast.decode': {'torch': {'status': 'candidate',
                                                               'implementation': 'hcu_adapter.kernels.mla_attention_emu.fused_decode',
                                                               'limitations': 'Torch sparse MLA decode on '
                                                                              'V4.1 FP8 (528B) / FP4 (288B) '
                                                                              'paged caches; FP32 O-RoPE '
                                                                              'through FP8 cast follows '
                                                                              'pinned CUDA finish_tile, not '
                                                                              "Python ref's intermediate "
                                                                              'BF16; pinned config only; '
                                                                              'candidate precision '
                                                                              'correction'},
                                                     'native': {'status': 'candidate',
                                                                'implementation': 'hcu_adapter.kernels.native_flashmla.fused_decode',
                                                                'limitations': 'Explicit code-object-v4 '
                                                                               '1287a01 plus optional FP32 '
                                                                               'API, gfx938 H64/H128 D512; '
                                                                               'LightOp Q/O-RoPE and FP8; '
                                                                               'decode gathers V4.1 packed '
                                                                               'cache to BF16, not native '
                                                                               'paged decode; qualification '
                                                                               'recorded separately'}},
 'flash_mla.fused_norm_rope_attn_rope_cast.prefill': {'torch': {'status': 'candidate',
                                                                'implementation': 'hcu_adapter.kernels.mla_attention_emu.fused_prefill',
                                                                'limitations': 'Torch sparse MLA prefill '
                                                                               'mirroring vendor ref (FP32 '
                                                                               'softmax, chunked gather); '
                                                                               'pinned V4.1 config only; '
                                                                               'correctness tier, no '
                                                                               'performance qualification'},
                                                      'native': {'status': 'candidate',
                                                                 'implementation': 'hcu_adapter.kernels.native_flashmla.fused_prefill',
                                                                 'limitations': 'Explicit code-object-v4 '
                                                                                '1287a01 plus optional FP32 '
                                                                                'API, gfx938 H64/H128 D512; '
                                                                                'LightOp Q/O-RoPE and FP8; '
                                                                                'decode gathers V4.1 packed '
                                                                                'cache to BF16, not native '
                                                                                'paged decode; qualification '
                                                                                'recorded separately'}},
 'tile_kernels.quant.swiglu_forward_and_per_token_cast': {'torch': {'status': 'candidate',
                                                                    'implementation': 'hcu_adapter.kernels.quant_emu.swiglu_forward_and_per_token_cast',
                                                                    'limitations': 'FP32 SwiGLU (+clamp, '
                                                                                   'topk weight) then '
                                                                                   'per-token cast; no clamp '
                                                                                   'counting; computes PSUM '
                                                                                   'gap rows (vendor leaves '
                                                                                   'them unspecified)'},
                                                          'lightop': {'status': 'candidate',
                                                                      'implementation': 'hcu_adapter.kernels.swiglu_lightop.swiglu_forward_and_per_token_cast',
                                                                      'limitations': 'Explicit LightOp '
                                                                                     'b2f8c9d; preserve '
                                                                                     'residual/clamp/FP32 '
                                                                                     'intermediate and SF '
                                                                                     'ABI; quant native only '
                                                                                     'rounded block32/128 '
                                                                                     'FP8, other formats '
                                                                                     'counted as '
                                                                                     'compatibility; '
                                                                                     'mega-MoE shared SF '
                                                                                     'unsupported; finite '
                                                                                     'and model '
                                                                                     'qualification recorded '
                                                                                     'separately'}},
 'tilelang_ops.compress_and_update_state_prefill_tl': {'triton': {'status': 'candidate',
                                                                  'implementation': 'hcu_adapter.kernels.compress_emu.compress_prefill',
                                                                  'limitations': 'Torch '
                                                                                 'gather/softmax/scatter on '
                                                                                 'device metadata '
                                                                                 '(graph-safe); no '
                                                                                 'model/performance '
                                                                                 'qualification'},
                                                       'torch': {'status': 'candidate',
                                                                 'implementation': 'hcu_adapter.kernels.compress_emu.compress_prefill',
                                                                 'limitations': 'Torch '
                                                                                'gather/softmax/scatter on '
                                                                                'device metadata '
                                                                                '(graph-safe); no '
                                                                                'model/performance '
                                                                                'qualification'}},
 'tilelang_ops.compress_and_update_state_decode_tl': {'triton': {'status': 'candidate',
                                                                 'implementation': 'hcu_adapter.kernels.compress_emu.compress_decode',
                                                                 'limitations': 'Torch '
                                                                                'gather/softmax/scatter on '
                                                                                'device metadata '
                                                                                '(graph-safe); no '
                                                                                'model/performance '
                                                                                'qualification'},
                                                      'torch': {'status': 'candidate',
                                                                'implementation': 'hcu_adapter.kernels.compress_emu.compress_decode',
                                                                'limitations': 'Torch gather/softmax/scatter '
                                                                               'on device metadata '
                                                                               '(graph-safe); no '
                                                                               'model/performance '
                                                                               'qualification'}},
 'tilelang_ops.load_kv_index_tl': {'triton': {'status': 'candidate',
                                              'implementation': 'hcu_adapter.kernels.index_ops.load_kv_index',
                                              'limitations': 'Torch index op mirroring the upstream *_ref '
                                                             '(device-only); no performance qualification'},
                                   'torch': {'status': 'candidate',
                                             'implementation': 'hcu_adapter.kernels.index_ops.load_kv_index',
                                             'limitations': 'Torch index op mirroring the upstream *_ref '
                                                            '(device-only); no performance qualification'}},
 'tilelang_ops.kv_index_cat_tl': {'triton': {'status': 'candidate',
                                             'implementation': 'hcu_adapter.kernels.index_ops.kv_index_cat',
                                             'limitations': 'Torch index op mirroring the upstream *_ref '
                                                            '(device-only); no performance qualification'},
                                  'torch': {'status': 'candidate',
                                            'implementation': 'hcu_adapter.kernels.index_ops.kv_index_cat',
                                            'limitations': 'Torch index op mirroring the upstream *_ref '
                                                           '(device-only); no performance qualification'}},
 'tilelang_ops.convert_kv_index_tl': {'triton': {'status': 'candidate',
                                                 'implementation': 'hcu_adapter.kernels.index_ops.convert_kv_index',
                                                 'limitations': 'Torch index op mirroring the upstream *_ref '
                                                                '(device-only); no performance '
                                                                'qualification'},
                                      'torch': {'status': 'candidate',
                                                'implementation': 'hcu_adapter.kernels.index_ops.convert_kv_index',
                                                'limitations': 'Torch index op mirroring the upstream *_ref '
                                                               '(device-only); no performance '
                                                               'qualification'}},
 'tilelang_ops.candidate_block_max_tl': {'triton': {'status': 'candidate',
                                                    'implementation': 'hcu_adapter.kernels.index_ops.candidate_block_max',
                                                    'limitations': 'Torch index op mirroring the upstream '
                                                                   '*_ref (device-only); no performance '
                                                                   'qualification'},
                                         'torch': {'status': 'candidate',
                                                   'implementation': 'hcu_adapter.kernels.index_ops.candidate_block_max',
                                                   'limitations': 'Torch index op mirroring the upstream '
                                                                  '*_ref (device-only); no performance '
                                                                  'qualification'}},
 'tilelang_ops.get_candidate_compact_lens_tl': {'triton': {'status': 'candidate',
                                                           'implementation': 'hcu_adapter.kernels.index_ops.candidate_compact_lens',
                                                           'limitations': 'Torch index op mirroring the '
                                                                          'upstream *_ref (device-only); no '
                                                                          'performance qualification'},
                                                'torch': {'status': 'candidate',
                                                          'implementation': 'hcu_adapter.kernels.index_ops.candidate_compact_lens',
                                                          'limitations': 'Torch index op mirroring the '
                                                                         'upstream *_ref (device-only); no '
                                                                         'performance qualification'}},
 'tilelang_ops.restore_candidate_indices_inplace_tl': {'triton': {'status': 'candidate',
                                                                  'implementation': 'hcu_adapter.kernels.index_ops.restore_candidate_indices',
                                                                  'limitations': 'Torch index op mirroring '
                                                                                 'the upstream *_ref '
                                                                                 '(device-only); no '
                                                                                 'performance qualification'},
                                                       'torch': {'status': 'candidate',
                                                                 'implementation': 'hcu_adapter.kernels.index_ops.restore_candidate_indices',
                                                                 'limitations': 'Torch index op mirroring '
                                                                                'the upstream *_ref '
                                                                                '(device-only); no '
                                                                                'performance qualification'}},
 'tile_kernels.quant.batched_transpose_weight_sf': {'torch': {'status': 'candidate',
                                                              'implementation': 'hcu_adapter.kernels.weight_scale_emu.batched_transpose_weight_sf',
                                                              'limitations': 'GPU UE8M0 int32, '
                                                                             'block32/block128, grouped '
                                                                             'layouts and reused outs; no '
                                                                             'model qualification'}},
 'tile_kernels.quant.per_token_cast': {'torch': {'status': 'candidate',
                                                 'implementation': 'hcu_adapter.kernels.quant_emu.per_token_cast',
                                                 'limitations': 'Deterministic BF16/FP32 to E4M3 or packed '
                                                                'E2M1, row/column SF and packed UE8M0; no '
                                                                'prequantized input, E4M3 scales, stochastic '
                                                                'cast or model qualification'},
                                       'lightop': {'status': 'candidate',
                                                   'implementation': 'hcu_adapter.kernels.quant_lightop.per_token_cast',
                                                   'limitations': 'Explicit LightOp b2f8c9d; preserve '
                                                                  'residual/clamp/FP32 intermediate and SF '
                                                                  'ABI; quant native only rounded '
                                                                  'block32/128 FP8, other formats counted as '
                                                                  'compatibility; mega-MoE shared SF '
                                                                  'unsupported; finite and model '
                                                                  'qualification recorded separately'}},
 'deep_gemm.bf16_gemm_nt': {'lightop': {'status': 'candidate',
                                        'implementation': 'hcu_adapter.kernels.gemm_lightop.bf16_gemm_nt',
                                        'limitations': 'Contiguous BF16 inputs, K divisible by 32; no e2e '
                                                       'qualification'}},
 'deep_gemm.transform_sf_into_required_layout': {'torch': {'status': 'candidate',
                                                           'implementation': 'hcu_adapter.kernels.weight_scale_emu.transform_sf_into_required_layout',
                                                           'limitations': 'SM100 packed UE8M0 ABI; no PSUM '
                                                                          'gaps or SM90 float-scale mode; no '
                                                                          'model qualification'}},
 'deep_gemm.set_num_sms': {'host': {'status': 'candidate',
                                    'implementation': 'hcu_adapter.kernels.runtime_config.deep_gemm_set_num_sms',
                                    'limitations': 'Host-side vendor knob mirrored from pinned DeepGEMM; '
                                                   'non-SM100 MK alignment 128; num_sms = HIP CU count'}},
 'deep_gemm.get_num_sms': {'host': {'status': 'candidate',
                                    'implementation': 'hcu_adapter.kernels.runtime_config.deep_gemm_get_num_sms',
                                    'limitations': 'Host-side vendor knob mirrored from pinned DeepGEMM; '
                                                   'non-SM100 MK alignment 128; num_sms = HIP CU count'}},
 'deep_gemm.set_mk_alignment_for_contiguous_layout': {'host': {'status': 'candidate',
                                                               'implementation': 'hcu_adapter.kernels.runtime_config.set_mk_alignment_for_contiguous_layout',
                                                               'limitations': 'Host-side vendor knob '
                                                                              'mirrored from pinned '
                                                                              'DeepGEMM; non-SM100 MK '
                                                                              'alignment 128; num_sms = HIP '
                                                                              'CU count'}},
 'deep_gemm.get_mk_alignment_for_contiguous_layout': {'host': {'status': 'candidate',
                                                               'implementation': 'hcu_adapter.kernels.runtime_config.get_mk_alignment_for_contiguous_layout',
                                                               'limitations': 'Host-side vendor knob '
                                                                              'mirrored from pinned '
                                                                              'DeepGEMM; non-SM100 MK '
                                                                              'alignment 128; num_sms = HIP '
                                                                              'CU count'}},
 'deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout': {'host': {'status': 'candidate',
                                                                           'implementation': 'hcu_adapter.kernels.runtime_config.get_theoretical_mk_alignment_for_contiguous_layout',
                                                                           'limitations': 'Host-side vendor '
                                                                                          'knob mirrored '
                                                                                          'from pinned '
                                                                                          'DeepGEMM; '
                                                                                          'non-SM100 MK '
                                                                                          'alignment 128; '
                                                                                          'num_sms = HIP CU '
                                                                                          'count'}},
 'deep_gemm.get_tma_aligned_size': {'host': {'status': 'candidate',
                                             'implementation': 'hcu_adapter.kernels.runtime_config.get_tma_aligned_size',
                                             'limitations': 'Host-side vendor knob mirrored from pinned '
                                                            'DeepGEMM; non-SM100 MK alignment 128; num_sms = '
                                                            'HIP CU count'}},
 'tile_kernels.set_num_sms': {'host': {'status': 'candidate',
                                       'implementation': 'hcu_adapter.kernels.runtime_config.tile_kernels_set_num_sms',
                                       'limitations': 'Host-side vendor knob mirrored from pinned '
                                                      'TileKernels 2.1.0 config.py; num_sms = HIP CU count'}},
 'tile_kernels.get_num_sms': {'host': {'status': 'candidate',
                                       'implementation': 'hcu_adapter.kernels.runtime_config.tile_kernels_get_num_sms',
                                       'limitations': 'Host-side vendor knob mirrored from pinned '
                                                      'TileKernels 2.1.0 config.py; num_sms = HIP CU count'}},
 'tile_kernels.set_token_alignment': {'host': {'status': 'candidate',
                                               'implementation': 'hcu_adapter.kernels.runtime_config.set_token_alignment',
                                               'limitations': 'Host-side vendor knob mirrored from pinned '
                                                              'TileKernels 2.1.0 config.py; num_sms = HIP CU '
                                                              'count'}},
 'tile_kernels.get_token_alignment': {'host': {'status': 'candidate',
                                               'implementation': 'hcu_adapter.kernels.runtime_config.get_token_alignment',
                                               'limitations': 'Host-side vendor knob mirrored from pinned '
                                                              'TileKernels 2.1.0 config.py; num_sms = HIP CU '
                                                              'count'}},
 'deep_gemm.fp8_fp4_gemm_nt': {'torch': {'status': 'candidate',
                                         'implementation': 'hcu_adapter.kernels.gemm_emu.fp8_fp4_gemm_nt',
                                         'limitations': 'FP8 E4M3 operands only; exact BF16/FP32 dequant + '
                                                        'torch GEMM (FP32 accumulate); no alpha/epilogue; '
                                                        'correctness tier, not performance'}},
 'deep_gemm.fp8_einsum': {'torch': {'status': 'candidate',
                                    'implementation': 'hcu_adapter.kernels.gemm_emu.fp8_einsum',
                                    'limitations': "'bhr,hdr->bhd' only; exact dequant + torch bmm; FP8-pair "
                                                   'output via per-32 UE8M0 cast of the BF16 result; '
                                                   'correctness tier'}},
 'deep_gemm.mega_mhc': {'torch': {'status': 'candidate',
                                  'implementation': 'hcu_adapter.kernels.mhc_emu.mega_mhc',
                                  'limitations': 'Composed torch mHC post+mixes+pre+RMSNorm(+per-32 FP8, '
                                                 'gemm/routed SF); no Mega MoE shared-SF layout; correctness '
                                                 'tier'},
                        'lightop': {'status': 'candidate',
                                    'implementation': 'hcu_adapter.kernels.mhc_lightop.mega_mhc',
                                    'limitations': 'Explicit LightOp b2f8c9d; preserve residual/clamp/FP32 '
                                                   'intermediate and SF ABI; quant native only rounded '
                                                   'block32/128 FP8, other formats counted as compatibility; '
                                                   'mega-MoE shared SF unsupported; finite and model '
                                                   'qualification recorded separately'}},
 'deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous': {'torch': {'status': 'candidate',
                                                              'implementation': 'hcu_adapter.kernels.grouped_gemm_emu.m_grouped_fp8_fp4_gemm_nt_contiguous',
                                                              'limitations': 'Route A correctness tier: '
                                                                             'exact BF16 dequant of FP8 acts '
                                                                             'x MXFP4/FP8 experts + dense '
                                                                             'per-group torch GEMM '
                                                                             '(graph-safe, O(G) work); DTK '
                                                                             'm_grouped_wfp4a16 rejected '
                                                                             '(calc_diff~1 on gfx938, see '
                                                                             'config.md)'},
                                                    'triton': {'status': 'candidate',
                                                               'implementation': 'hcu_adapter.kernels.grouped_gemm_triton.m_grouped_fp8_fp4_gemm_nt_contiguous',
                                                               'limitations': 'Route A fast tier: '
                                                                              'per-BLOCK_M group lookup, '
                                                                              'FP8/E2M1 x UE8M0 dequantized '
                                                                              'to BF16 in registers, FP32 '
                                                                              'accumulate (same products as '
                                                                              'the torch tier); per-32 '
                                                                              'packed UE8M0 recipes only, '
                                                                              'others use the torch tier; '
                                                                              'tiles from a BW1101 sweep '
                                                                              '(tests/tools/_probe_grouped_gemm_bench.py)'}},
 'deep_gemm.fp8_fp4_mqa_logits': {'torch': {'status': 'candidate',
                                            'implementation': 'hcu_adapter.kernels.mla_attention_emu.fp8_fp4_mqa_logits',
                                            'limitations': 'Dense prefill indexer logits (compressed layout, '
                                                           'BF16, tail zeroed); FP8/FP4 with per-row MX '
                                                           'scales; KV-chunked torch; correctness tier'},
                                  'lightop': {'status': 'candidate',
                                              'implementation': 'hcu_adapter.kernels.mqa_logits_lightop.fp8_fp4_mqa_logits',
                                              'limitations': 'D128 H8/16/32 MXFP4 uses packed LightOp '
                                                             'sparse/paged math; unpaged sparse K is '
                                                             'byte-repaged for unaligned request starts, not '
                                                             'dequantized per query; paged dense requires '
                                                             'page64; BF16 logits default/zero tail '
                                                             'retained; explicit output_dtype supports the '
                                                             'independent plugin FP32 policy; other layouts '
                                                             'use counted compatibility bridge; '
                                                             'qualification recorded separately'}},
 'deep_gemm.fp8_fp4_sparse_mqa_logits': {'torch': {'status': 'candidate',
                                                   'implementation': 'hcu_adapter.kernels.mla_attention_emu.fp8_fp4_sparse_mqa_logits',
                                                   'limitations': 'Sparse (candidate-block) indexer logits '
                                                                  'per vendor test contract; metadata is an '
                                                                  'HCU-private dict; correctness tier'},
                                         'lightop': {'status': 'candidate',
                                                     'implementation': 'hcu_adapter.kernels.mqa_logits_lightop.fp8_fp4_sparse_mqa_logits',
                                                     'limitations': 'D128 H8/16/32 MXFP4 uses packed LightOp '
                                                                    'sparse/paged math; unpaged sparse K is '
                                                                    'byte-repaged for unaligned request '
                                                                    'starts, not dequantized per query; '
                                                                    'paged dense requires page64; BF16 '
                                                                    'logits default/zero tail retained; '
                                                                    'explicit output_dtype supports the '
                                                                    'independent plugin FP32 policy; other '
                                                                    'layouts use counted compatibility '
                                                                    'bridge; qualification recorded '
                                                                    'separately'}},
 'deep_gemm.get_sparse_mqa_logits_metadata': {'torch': {'status': 'candidate',
                                                        'implementation': 'hcu_adapter.kernels.mla_attention_emu.get_sparse_mqa_logits_metadata',
                                                        'limitations': 'Sparse (candidate-block) indexer '
                                                                       'logits per vendor test contract; '
                                                                       'metadata is an HCU-private dict; '
                                                                       'correctness tier'}},
 'deep_gemm.fp8_fp4_paged_mqa_logits': {'torch': {'status': 'candidate',
                                                  'implementation': 'hcu_adapter.kernels.mla_attention_emu.fp8_fp4_paged_mqa_logits',
                                                  'limitations': 'Sparse (candidate-block) indexer logits '
                                                                 'per vendor test contract; metadata is an '
                                                                 'HCU-private dict; correctness tier'},
                                        'lightop': {'status': 'candidate',
                                                    'implementation': 'hcu_adapter.kernels.mqa_logits_lightop.fp8_fp4_paged_mqa_logits',
                                                    'limitations': 'D128 H8/16/32 MXFP4 uses packed LightOp '
                                                                   'sparse/paged math; unpaged sparse K is '
                                                                   'byte-repaged for unaligned request '
                                                                   'starts, not dequantized per query; paged '
                                                                   'dense requires page64; BF16 logits '
                                                                   'default/zero tail retained; explicit '
                                                                   'output_dtype supports the independent '
                                                                   'plugin FP32 policy; other layouts use '
                                                                   'counted compatibility bridge; '
                                                                   'qualification recorded separately'}},
 'deep_gemm.get_paged_mqa_logits_metadata': {'torch': {'status': 'candidate',
                                                       'implementation': 'hcu_adapter.kernels.mla_attention_emu.get_paged_mqa_logits_metadata',
                                                       'limitations': 'Sparse (candidate-block) indexer '
                                                                      'logits per vendor test contract; '
                                                                      'metadata is an HCU-private dict; '
                                                                      'correctness tier'}},
 'deep_gemm.fp8_fp4_paged_sparse_mqa_logits': {'torch': {'status': 'candidate',
                                                         'implementation': 'hcu_adapter.kernels.mla_attention_emu.fp8_fp4_paged_sparse_mqa_logits',
                                                         'limitations': 'Sparse (candidate-block) indexer '
                                                                        'logits per vendor test contract; '
                                                                        'metadata is an HCU-private dict; '
                                                                        'correctness tier'},
                                               'lightop': {'status': 'candidate',
                                                           'implementation': 'hcu_adapter.kernels.mqa_logits_lightop.fp8_fp4_paged_sparse_mqa_logits',
                                                           'limitations': 'D128 H8/16/32 MXFP4 uses packed '
                                                                          'LightOp sparse/paged math; '
                                                                          'unpaged sparse K is byte-repaged '
                                                                          'for unaligned request starts, not '
                                                                          'dequantized per query; paged '
                                                                          'dense requires page64; BF16 '
                                                                          'logits default/zero tail '
                                                                          'retained; explicit output_dtype '
                                                                          'supports the independent plugin '
                                                                          'FP32 policy; other layouts use '
                                                                          'counted compatibility bridge; '
                                                                          'qualification recorded '
                                                                          'separately'}},
 'deep_gemm.get_paged_sparse_mqa_logits_metadata': {'torch': {'status': 'candidate',
                                                              'implementation': 'hcu_adapter.kernels.mla_attention_emu.get_paged_sparse_mqa_logits_metadata',
                                                              'limitations': 'Sparse (candidate-block) '
                                                                             'indexer logits per vendor test '
                                                                             'contract; metadata is an '
                                                                             'HCU-private dict; correctness '
                                                                             'tier'}},
 'deep_gemm.bf16_mega_gate': {'torch': {'status': 'candidate',
                                        'implementation': 'hcu_adapter.kernels.moe_gate_emu.bf16_mega_gate',
                                        'limitations': 'FP32 gate GEMM + graph-safe sqrtsoftplus stable '
                                                       'top-k (tile_kernels reference); no force_random; '
                                                       'correctness tier'},
                              'lightop': {'status': 'candidate',
                                          'implementation': 'hcu_adapter.kernels.moe_gate_lightop.bf16_mega_gate',
                                          'limitations': 'Explicit LightOp b2f8c9d; preserve '
                                                         'residual/clamp/FP32 intermediate and SF ABI; quant '
                                                         'native only rounded block32/128 FP8, other formats '
                                                         'counted as compatibility; mega-MoE shared SF '
                                                         'unsupported; finite and model qualification '
                                                         'recorded separately; 256/384 expert routing uses '
                                                         'LightOp; 128/top3 draft uses SGLang-aligned Triton '
                                                         'log1p/NaN-first/lowest-id ties; force_random '
                                                         'metadata retains existing HCU normal routing; '
                                                         'forced-balance benchmark remains guarded'}},
 'tile_kernels.moe.moe_topk_gate': {'torch': {'status': 'candidate',
                                              'implementation': 'hcu_adapter.kernels.moe_gate_emu.moe_topk_gate',
                                              'limitations': 'Graph-safe rewrite of tile_kernels torch '
                                                             'reference; no force_random or training scores '
                                                             'output'},
                                    'lightop': {'status': 'candidate',
                                                'implementation': 'hcu_adapter.kernels.moe_gate_lightop.moe_topk_gate',
                                                'limitations': 'Explicit LightOp b2f8c9d; preserve '
                                                               'residual/clamp/FP32 intermediate and SF ABI; '
                                                               'quant native only rounded block32/128 FP8, '
                                                               'other formats counted as compatibility; '
                                                               'mega-MoE shared SF unsupported; finite and '
                                                               'model qualification recorded separately; '
                                                               '256/384 expert routing uses LightOp; '
                                                               '128/top3 draft uses SGLang-aligned Triton '
                                                               'log1p/NaN-first/lowest-id ties; force_random '
                                                               'metadata retains existing HCU normal '
                                                               'routing; forced-balance benchmark remains '
                                                               'guarded'}},
 'deep_gemm.tf32_hc_prenorm_gemm': {'torch': {'status': 'candidate',
                                              'implementation': 'hcu_adapter.kernels.mhc_emu.tf32_hc_prenorm_gemm',
                                              'limitations': 'BF16 x FP32 prenorm GEMM, contiguous K-major '
                                                             'contract, K tiled in 64-element blocks; no '
                                                             'model or performance qualification'}},
 'tile_kernels.mhc.expand_kernel.expand_to_mhc_fwd': {'torch': {'status': 'candidate',
                                                                'implementation': 'hcu_adapter.kernels.mhc_emu.expand_to_mhc_fwd',
                                                                'limitations': 'Pinned mHC forward contract; '
                                                                               'eager candidate, no model or '
                                                                               'performance qualification'}},
 'tile_kernels.mhc.pre_apply_mix_kernel.mhc_pre_apply_mix_fwd': {'torch': {'status': 'candidate',
                                                                           'implementation': 'hcu_adapter.kernels.mhc_emu.mhc_pre_apply_mix_fwd',
                                                                           'limitations': 'Pinned mHC '
                                                                                          'forward contract; '
                                                                                          'eager candidate, '
                                                                                          'no model or '
                                                                                          'performance '
                                                                                          'qualification'}},
 'tile_kernels.mhc.head_compute_mix_kernel.mhc_head_compute_mix_fwd': {'torch': {'status': 'candidate',
                                                                                 'implementation': 'hcu_adapter.kernels.mhc_emu.mhc_head_compute_mix_fwd',
                                                                                 'limitations': 'Pinned mHC '
                                                                                                'forward '
                                                                                                'contract; '
                                                                                                'eager '
                                                                                                'candidate, '
                                                                                                'no model or '
                                                                                                'performance '
                                                                                                'qualification'}},
 'tile_kernels.mhc.pre_split_mixes_kernel.mhc_pre_split_mixes_fwd': {'torch': {'status': 'candidate',
                                                                               'implementation': 'hcu_adapter.kernels.mhc_emu.mhc_pre_split_mixes_fwd',
                                                                               'limitations': 'Pinned mHC '
                                                                                              'forward '
                                                                                              'contract; '
                                                                                              'eager '
                                                                                              'candidate, no '
                                                                                              'model or '
                                                                                              'performance '
                                                                                              'qualification'}},
 'tile_kernels.mhc.sinkhorn_kernel.mhc_sinkhorn_fwd': {'torch': {'status': 'candidate',
                                                                 'implementation': 'hcu_adapter.kernels.mhc_emu.mhc_sinkhorn_fwd',
                                                                 'limitations': 'Pinned mHC forward '
                                                                                'contract; eager candidate, '
                                                                                'no model or performance '
                                                                                'qualification'}},
 'tile_kernels.mhc.post_kernel.mhc_post_fwd': {'torch': {'status': 'candidate',
                                                         'implementation': 'hcu_adapter.kernels.mhc_emu.mhc_post_fwd',
                                                         'limitations': 'Pinned mHC forward contract; eager '
                                                                        'candidate, no model or performance '
                                                                        'qualification'}},
 'tile_kernels.mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd': {'torch': {'status': 'candidate',
                                                                      'implementation': 'hcu_adapter.kernels.mhc_emu.mhc_fn_normw_merge_fwd',
                                                                      'limitations': 'Pinned mHC forward '
                                                                                     'contract; eager '
                                                                                     'candidate, no model or '
                                                                                     'performance '
                                                                                     'qualification'}},
 'tile_kernels.mhc.norm_fn_kernel.mhc_reduce_partials_and_rmsnorm_fwd': {'torch': {'status': 'candidate',
                                                                                   'implementation': 'hcu_adapter.kernels.mhc_emu.mhc_reduce_partials_and_rmsnorm_fwd',
                                                                                   'limitations': 'Pinned '
                                                                                                  'mHC '
                                                                                                  'forward '
                                                                                                  'contract; '
                                                                                                  'eager '
                                                                                                  'candidate, '
                                                                                                  'no model '
                                                                                                  'or '
                                                                                                  'performance '
                                                                                                  'qualification'}},
 'tile_kernels.mhc.pre_big_fuse_kernel.mhc_pre_big_fuse_fwd': {'torch': {'status': 'candidate',
                                                                         'implementation': 'hcu_adapter.kernels.mhc_emu.mhc_pre_big_fuse_fwd',
                                                                         'limitations': 'Pinned mHC forward '
                                                                                        'contract; eager '
                                                                                        'candidate, no model '
                                                                                        'or performance '
                                                                                        'qualification'}},
 'flash_mla.fused_norm_rope_attn_rope_cast.permute_q_b_proj': {'torch': {'status': 'candidate',
                                                                         'implementation': 'hcu_adapter.kernels.mla_layout_emu.permute_q_b_proj',
                                                                         'limitations': 'Byte-preserving '
                                                                                        'initialization '
                                                                                        'permutation only; '
                                                                                        'no fused attention '
                                                                                        'or model '
                                                                                        'qualification'}},
 'flash_mla.fused_norm_rope_attn_rope_cast.permute_wv_proj': {'torch': {'status': 'candidate',
                                                                        'implementation': 'hcu_adapter.kernels.mla_layout_emu.permute_wv_proj',
                                                                        'limitations': 'Byte-preserving '
                                                                                       'initialization '
                                                                                       'permutation only; no '
                                                                                       'fused attention or '
                                                                                       'model '
                                                                                       'qualification'}}}
