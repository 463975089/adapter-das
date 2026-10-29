"""DeepInfer upstream aux interface providers; installed by the host bridge."""

BACKEND_CATALOG = {'deep_ep.BucketBuffer': {'rccl': {'status': 'candidate',
                                   'implementation': 'hcu_adapter.kernels.bucket.BucketBuffer',
                                   'limitations': 'RCCL session storage with pinned 2MiB/64B alignments; '
                                                  'byte-preserving CP all-gather; numerical/graph/CP model '
                                                  'qualification pending; no symmetric-memory registration'}},
 'deep_ep.BucketSession': {'rccl': {'status': 'candidate',
                                    'implementation': 'hcu_adapter.kernels.bucket.BucketSession',
                                    'limitations': 'RCCL session storage with pinned 2MiB/64B alignments; '
                                                   'byte-preserving CP all-gather; numerical/graph/CP model '
                                                   'qualification pending; no symmetric-memory '
                                                   'registration'}},
 'deep_ep.get_num_allocation_alignment': {'rccl': {'status': 'candidate',
                                                   'implementation': 'hcu_adapter.kernels.bucket.get_num_allocation_alignment',
                                                   'limitations': 'RCCL session storage with pinned 2MiB/64B '
                                                                  'alignments; byte-preserving CP '
                                                                  'all-gather; numerical/graph/CP model '
                                                                  'qualification pending; no '
                                                                  'symmetric-memory registration'}},
 'deep_ep.get_num_rdma_alignment': {'rccl': {'status': 'candidate',
                                             'implementation': 'hcu_adapter.kernels.bucket.get_num_rdma_alignment',
                                             'limitations': 'RCCL session storage with pinned 2MiB/64B '
                                                            'alignments; byte-preserving CP all-gather; '
                                                            'numerical/graph/CP model qualification pending; '
                                                            'no symmetric-memory registration'}},
 'flash_attn.cute.interface._flash_attn_fwd': {'torch': {'status': 'candidate',
                                                         'implementation': 'hcu_adapter.kernels.vision_attention_emu.flash_attn_fwd',
                                                         'limitations': 'Pinned ViT noncausal varlen Q/K/V '
                                                                        'call; bounded FP32 query tiles; '
                                                                        'numerical/graph/model qualification '
                                                                        'pending'}},
 'deep_ep.EngramBuffer': {'rccl_gpu': {'status': 'candidate',
                                       'implementation': 'hcu_adapter.kernels.deepep_compat.EngramBuffer',
                                       'limitations': 'GPU tables, eager, byte-exact collective lookup; no '
                                                      'CPU RDMA/graph/e2e qualification'},
                          'rccl_gpu_graph': {'status': 'candidate',
                                             'implementation': 'hcu_adapter.kernels.deepep_compat.GraphEngramBuffer',
                                             'limitations': 'Explicit graph candidate for GPU tables and '
                                                            'fixed query capacity; 8-rank BF16/FP8 '
                                                            'changed-query replay and model smoke '
                                                            '(attempt44) passed; formal '
                                                            'accuracy/long-context pending, no CPU RDMA'}},
 'deep_ep.EPBuffer': {'dtk_v1_expanded': {'status': 'candidate',
                                          'implementation': 'hcu_adapter.kernels.deepep_compat.EPBuffer',
                                          'limitations': 'Single-host eager expanded dispatch; no graph, '
                                                         'cached dispatch, EPLB or e2e qualification'},
                      'dtk_v1_expanded_graph': {'status': 'candidate',
                                                'implementation': 'hcu_adapter.kernels.deepep_compat.GraphEPBuffer',
                                                'limitations': 'Explicit single-host graph candidate; fixed '
                                                               'receive bound and do_cpu_sync=False during '
                                                               'capture; largest decode warmup must create '
                                                               'the separate fixed-capacity Buffer before '
                                                               'capture; 8-rank mixed graph/prefill and '
                                                               'model smoke (attempt44) passed; formal '
                                                               'accuracy/long-context pending; no cached '
                                                               'dispatch/EPLB'}},
 'deep_ep.get_physical_domain_size': {'lsa1': {'status': 'candidate',
                                               'implementation': 'hcu_adapter.kernels.deepep_compat.get_physical_domain_size',
                                               'limitations': 'Reports LSA domain 1 on DTK2610 (no RCCL LSA '
                                                              'team, R-16 VMM cap); routes MoE off MegaMoE '
                                                              'and EPLB off intranode symm-mem shuffle'}},
 'deep_ep.get_logical_domain_size': {'lsa1': {'status': 'candidate',
                                              'implementation': 'hcu_adapter.kernels.deepep_compat.get_logical_domain_size',
                                              'limitations': 'Reports LSA domain 1 on DTK2610 (no RCCL LSA '
                                                             'team, R-16 VMM cap); routes MoE off MegaMoE '
                                                             'and EPLB off intranode symm-mem shuffle'}},
 'tilelang_ops.fused_sample_tl': {'triton': {'status': 'candidate',
                                             'implementation': 'hcu_adapter.kernels.sampling_emu.fused_sample',
                                             'limitations': 'Mirrors upstream '
                                                            'fused_sample_ref/random_sampling_ref (torch '
                                                            'sort/cumsum, SplitMix64); property-equivalent '
                                                            'to the TileLang kernel, not bitwise; no '
                                                            'performance qualification'},
                                  'torch': {'status': 'candidate',
                                            'implementation': 'hcu_adapter.kernels.sampling_emu.fused_sample',
                                            'limitations': 'Mirrors upstream '
                                                           'fused_sample_ref/random_sampling_ref (torch '
                                                           'sort/cumsum, SplitMix64); property-equivalent to '
                                                           'the TileLang kernel, not bitwise; no performance '
                                                           'qualification'}},
 'tilelang_ops.random_sampling_tl': {'triton': {'status': 'candidate',
                                                'implementation': 'hcu_adapter.kernels.sampling_emu.random_sampling',
                                                'limitations': 'Mirrors upstream '
                                                               'fused_sample_ref/random_sampling_ref (torch '
                                                               'sort/cumsum, SplitMix64); '
                                                               'property-equivalent to the TileLang kernel, '
                                                               'not bitwise; no performance qualification'},
                                     'torch': {'status': 'candidate',
                                               'implementation': 'hcu_adapter.kernels.sampling_emu.random_sampling',
                                               'limitations': 'Mirrors upstream '
                                                              'fused_sample_ref/random_sampling_ref (torch '
                                                              'sort/cumsum, SplitMix64); property-equivalent '
                                                              'to the TileLang kernel, not bitwise; no '
                                                              'performance qualification'}},
 'tilelang_ops.prepare_v4_decode_metadata_tl': {'triton': {'status': 'candidate',
                                                           'implementation': 'hcu_adapter.kernels.zos_metadata_triton.prepare_v4_decode_metadata',
                                                           'limitations': 'Fused device-only Triton '
                                                                          'metadata; 50 exact '
                                                                          'integer/changed-input Graph cases '
                                                                          'and V4.1 CP8EP8/DP8EP8 DSpark '
                                                                          'regression passed (2026-10-28); '
                                                                          'pinned layouts only'},
                                                'torch': {'status': 'candidate',
                                                          'implementation': 'hcu_adapter.kernels.zos_metadata_emu.prepare_v4_decode_metadata',
                                                          'limitations': 'Device-only torch reference for '
                                                                         'decode/ZOS metadata'}},
 'tilelang_ops.copy_block_tables_tl': {'triton': {'status': 'candidate',
                                                  'implementation': 'hcu_adapter.kernels.zos_metadata_triton.copy_block_tables',
                                                  'limitations': 'Fused device-only Triton metadata; 50 '
                                                                 'exact integer/changed-input Graph cases '
                                                                 'and V4.1 CP8EP8/DP8EP8 DSpark regression '
                                                                 'passed (2026-10-28); pinned layouts only'},
                                       'torch': {'status': 'candidate',
                                                 'implementation': 'hcu_adapter.kernels.zos_metadata_emu.copy_block_tables',
                                                 'limitations': 'Device-only torch reference for decode/ZOS '
                                                                'metadata'}},
 'tilelang_ops.gather_input_ids_zos_tl': {'triton': {'status': 'candidate',
                                                     'implementation': 'hcu_adapter.kernels.zos_metadata_triton.gather_input_ids',
                                                     'limitations': 'Fused device-only Triton metadata; 50 '
                                                                    'exact integer/changed-input Graph cases '
                                                                    'and V4.1 CP8EP8/DP8EP8 DSpark '
                                                                    'regression passed (2026-10-28); pinned '
                                                                    'layouts only'},
                                          'torch': {'status': 'candidate',
                                                    'implementation': 'hcu_adapter.kernels.zos_metadata_emu.gather_input_ids',
                                                    'limitations': 'Device-only torch reference for '
                                                                   'decode/ZOS metadata'}},
 'tilelang_ops.gather_engram_tokens_zos_tl': {'triton': {'status': 'candidate',
                                                         'implementation': 'hcu_adapter.kernels.zos_metadata_triton.gather_engram_tokens',
                                                         'limitations': 'Fused device-only Triton metadata; '
                                                                        '50 exact integer/changed-input '
                                                                        'Graph cases and V4.1 CP8EP8/DP8EP8 '
                                                                        'DSpark regression passed '
                                                                        '(2026-10-28); pinned layouts only'},
                                              'torch': {'status': 'candidate',
                                                        'implementation': 'hcu_adapter.kernels.zos_metadata_emu.gather_engram_tokens',
                                                        'limitations': 'Device-only torch reference for '
                                                                       'decode/ZOS metadata'}},
 'tilelang_ops.compute_seqlens_k_zos_tl': {'triton': {'status': 'candidate',
                                                      'implementation': 'hcu_adapter.kernels.zos_metadata_triton.compute_seqlens_k',
                                                      'limitations': 'Fused device-only Triton metadata; 50 '
                                                                     'exact integer/changed-input Graph '
                                                                     'cases and V4.1 CP8EP8/DP8EP8 DSpark '
                                                                     'regression passed (2026-10-28); pinned '
                                                                     'layouts only'},
                                           'torch': {'status': 'candidate',
                                                     'implementation': 'hcu_adapter.kernels.zos_metadata_emu.compute_seqlens_k',
                                                     'limitations': 'Device-only torch reference for '
                                                                    'decode/ZOS metadata'}},
 'tilelang_ops.prepare_verification_metadata_tl': {'triton': {'status': 'candidate',
                                                              'implementation': 'hcu_adapter.kernels.verification.prepare_verification_metadata',
                                                              'limitations': 'Device-only torch mirror of '
                                                                             'the upstream sampler *_ref '
                                                                             '(scatter instead of mask '
                                                                             'indexing); no performance '
                                                                             'qualification'},
                                                   'torch': {'status': 'candidate',
                                                             'implementation': 'hcu_adapter.kernels.verification.prepare_verification_metadata',
                                                             'limitations': 'Device-only torch mirror of the '
                                                                            'upstream sampler *_ref (scatter '
                                                                            'instead of mask indexing); no '
                                                                            'performance qualification'}},
 'tilelang_ops.expand_sampling_params_tl': {'triton': {'status': 'candidate',
                                                       'implementation': 'hcu_adapter.kernels.verification.expand_sampling_params',
                                                       'limitations': 'Device-only torch mirror of the '
                                                                      'upstream sampler *_ref (scatter '
                                                                      'instead of mask indexing); no '
                                                                      'performance qualification'},
                                            'torch': {'status': 'candidate',
                                                      'implementation': 'hcu_adapter.kernels.verification.expand_sampling_params',
                                                      'limitations': 'Device-only torch mirror of the '
                                                                     'upstream sampler *_ref (scatter '
                                                                     'instead of mask indexing); no '
                                                                     'performance qualification'}},
 'tilelang_ops.map_lookup_tl': {'triton': {'status': 'candidate',
                                           'implementation': 'hcu_adapter.kernels.verification.map_lookup',
                                           'limitations': 'Device-only torch mirror of the upstream sampler '
                                                          '*_ref (scatter instead of mask indexing); no '
                                                          'performance qualification'},
                                'torch': {'status': 'candidate',
                                          'implementation': 'hcu_adapter.kernels.verification.map_lookup',
                                          'limitations': 'Device-only torch mirror of the upstream sampler '
                                                         '*_ref (scatter instead of mask indexing); no '
                                                         'performance qualification'}},
 'tilelang_ops.remap_draft_topk_tl': {'triton': {'status': 'candidate',
                                                 'implementation': 'hcu_adapter.kernels.verification.remap_draft_topk',
                                                 'limitations': 'Device-only torch mirror of the upstream '
                                                                'sampler *_ref (scatter instead of mask '
                                                                'indexing); no performance qualification'},
                                      'torch': {'status': 'candidate',
                                                'implementation': 'hcu_adapter.kernels.verification.remap_draft_topk',
                                                'limitations': 'Device-only torch mirror of the upstream '
                                                               'sampler *_ref (scatter instead of mask '
                                                               'indexing); no performance qualification'}},
 'tilelang_ops.get_accept_result_tl': {'triton': {'status': 'candidate',
                                                  'implementation': 'hcu_adapter.kernels.verification.get_accept_result',
                                                  'limitations': 'Device-only torch mirror of the upstream '
                                                                 'sampler *_ref (scatter instead of mask '
                                                                 'indexing); no performance qualification'},
                                       'torch': {'status': 'candidate',
                                                 'implementation': 'hcu_adapter.kernels.verification.get_accept_result',
                                                 'limitations': 'Device-only torch mirror of the upstream '
                                                                'sampler *_ref (scatter instead of mask '
                                                                'indexing); no performance qualification'}},
 'tilelang_ops.finalize_verification_tl': {'triton': {'status': 'candidate',
                                                      'implementation': 'hcu_adapter.kernels.verification.finalize_verification',
                                                      'limitations': 'Device-only torch mirror of the '
                                                                     'upstream sampler *_ref (scatter '
                                                                     'instead of mask indexing); no '
                                                                     'performance qualification'},
                                           'torch': {'status': 'candidate',
                                                     'implementation': 'hcu_adapter.kernels.verification.finalize_verification',
                                                     'limitations': 'Device-only torch mirror of the '
                                                                    'upstream sampler *_ref (scatter instead '
                                                                    'of mask indexing); no performance '
                                                                    'qualification'}},
 'tilelang_ops.sum_smaller_probs_tl': {'triton': {'status': 'candidate',
                                                  'implementation': 'hcu_adapter.kernels.verification.sum_smaller_probs',
                                                  'limitations': 'Device-only torch mirror of the upstream '
                                                                 'sampler *_ref (scatter instead of mask '
                                                                 'indexing); no performance qualification'},
                                       'torch': {'status': 'candidate',
                                                 'implementation': 'hcu_adapter.kernels.verification.sum_smaller_probs',
                                                 'limitations': 'Device-only torch mirror of the upstream '
                                                                'sampler *_ref (scatter instead of mask '
                                                                'indexing); no performance qualification'}},
 'tile_kernels.engram.fused_weight': {'torch': {'status': 'candidate',
                                                'implementation': 'hcu_adapter.kernels.engram_init_emu.fused_weight',
                                                'limitations': 'Pinned initialization contract; no model/e2e '
                                                               'or throughput qualification'}},
 'tile_kernels.torch.engram.make_offsets': {'torch': {'status': 'candidate',
                                                      'implementation': 'hcu_adapter.kernels.engram_init_emu.make_offsets',
                                                      'limitations': 'Pinned initialization contract; no '
                                                                     'model/e2e or throughput '
                                                                     'qualification'}},
 'tile_kernels.engram.engram_gate_fwd': {'triton': {'status': 'candidate',
                                                    'implementation': 'hcu_adapter.kernels.engram_runtime_emu.engram_gate_fwd',
                                                    'limitations': 'SGLang 7b91060 fused inference gate with '
                                                                   'pre-fused FP32 weights; saved backward '
                                                                   'intermediates retain torch reference; '
                                                                   'model qualification recorded '
                                                                   'separately'}},
 'tile_kernels.engram.engram_hash': {'torch': {'status': 'candidate',
                                               'implementation': 'hcu_adapter.kernels.engram_runtime_emu.engram_hash',
                                               'limitations': 'Hash requires nonnegative INT64-safe unmasked '
                                                              'inputs; model qualification recorded '
                                                              'separately'}},
 'tilelang_ops.confidence_head_fused_tl': {'triton': {'status': 'candidate',
                                                      'implementation': 'hcu_adapter.kernels.mtp.confidence_head',
                                                      'limitations': 'Device-only MTP/state/constraint '
                                                                     'contract; affected upstream refs and '
                                                                     'changed-input graph passed; no full '
                                                                     'feature qualification'},
                                           'torch': {'status': 'candidate',
                                                     'implementation': 'hcu_adapter.kernels.mtp.confidence_head',
                                                     'limitations': 'Device-only MTP/state/constraint '
                                                                    'contract; affected upstream refs and '
                                                                    'changed-input graph passed; no full '
                                                                    'feature qualification'}},
 'tilelang_ops.dflash_input_store_tl': {'triton': {'status': 'candidate',
                                                   'implementation': 'hcu_adapter.kernels.mtp.dflash_input_store',
                                                   'limitations': 'Device-only MTP/state/constraint '
                                                                  'contract; affected upstream refs and '
                                                                  'changed-input graph passed; no full '
                                                                  'feature qualification'},
                                        'torch': {'status': 'candidate',
                                                  'implementation': 'hcu_adapter.kernels.mtp.dflash_input_store',
                                                  'limitations': 'Device-only MTP/state/constraint contract; '
                                                                 'affected upstream refs and changed-input '
                                                                 'graph passed; no full feature '
                                                                 'qualification'}},
 'tilelang_ops.copy_dflash_decode_input_tl': {'triton': {'status': 'candidate',
                                                         'implementation': 'hcu_adapter.kernels.mtp.copy_decode_input',
                                                         'limitations': 'Device-only MTP/state/constraint '
                                                                        'contract; affected upstream refs '
                                                                        'and changed-input graph passed; no '
                                                                        'full feature qualification'},
                                              'torch': {'status': 'candidate',
                                                        'implementation': 'hcu_adapter.kernels.mtp.copy_decode_input',
                                                        'limitations': 'Device-only MTP/state/constraint '
                                                                       'contract; affected upstream refs and '
                                                                       'changed-input graph passed; no full '
                                                                       'feature qualification'}},
 'tilelang_ops.store_mtp_info_tl': {'triton': {'status': 'candidate',
                                               'implementation': 'hcu_adapter.kernels.mtp.store_proposals',
                                               'limitations': 'Device-only MTP/state/constraint contract; '
                                                              'affected upstream refs and changed-input '
                                                              'graph passed; no full feature qualification'},
                                    'torch': {'status': 'candidate',
                                              'implementation': 'hcu_adapter.kernels.mtp.store_proposals',
                                              'limitations': 'Device-only MTP/state/constraint contract; '
                                                             'affected upstream refs and changed-input graph '
                                                             'passed; no full feature qualification'}},
 'tilelang_ops.compute_seqlens_q_by_confidence_rates_tl': {'triton': {'status': 'candidate',
                                                                      'implementation': 'hcu_adapter.kernels.mtp.schedule_by_confidence',
                                                                      'limitations': 'Device-only '
                                                                                     'MTP/state/constraint '
                                                                                     'contract; affected '
                                                                                     'upstream refs and '
                                                                                     'changed-input graph '
                                                                                     'passed; no full '
                                                                                     'feature qualification'},
                                                           'torch': {'status': 'candidate',
                                                                     'implementation': 'hcu_adapter.kernels.mtp.schedule_by_confidence',
                                                                     'limitations': 'Device-only '
                                                                                    'MTP/state/constraint '
                                                                                    'contract; affected '
                                                                                    'upstream refs and '
                                                                                    'changed-input graph '
                                                                                    'passed; no full feature '
                                                                                    'qualification'}},
 'tilelang_ops.get_fused_dflash_get_input_tl': {'triton': {'status': 'candidate',
                                                           'implementation': 'hcu_adapter.kernels.mtp.dflash_get_input',
                                                           'limitations': 'Device-only MTP/state/constraint '
                                                                          'contract; affected upstream refs '
                                                                          'and changed-input graph passed; '
                                                                          'no full feature qualification'},
                                                'torch': {'status': 'candidate',
                                                          'implementation': 'hcu_adapter.kernels.mtp.dflash_get_input',
                                                          'limitations': 'Device-only MTP/state/constraint '
                                                                         'contract; affected upstream refs '
                                                                         'and changed-input graph passed; no '
                                                                         'full feature qualification'}},
 'tilelang_ops.select_state_slice_tl': {'triton': {'status': 'candidate',
                                                   'implementation': 'hcu_adapter.kernels.state_and_constraints.select_state_slice',
                                                   'limitations': 'Device-only MTP/state/constraint '
                                                                  'contract; affected upstream refs and '
                                                                  'changed-input graph passed; no full '
                                                                  'feature qualification'},
                                        'torch': {'status': 'candidate',
                                                  'implementation': 'hcu_adapter.kernels.state_and_constraints.select_state_slice',
                                                  'limitations': 'Device-only MTP/state/constraint contract; '
                                                                 'affected upstream refs and changed-input '
                                                                 'graph passed; no full feature '
                                                                 'qualification'}},
 'tilelang_ops.load_state_to_cache_tl': {'triton': {'status': 'candidate',
                                                    'implementation': 'hcu_adapter.kernels.state_and_constraints.load_state_to_cache',
                                                    'limitations': 'Device-only MTP/state/constraint '
                                                                   'contract; affected upstream refs and '
                                                                   'changed-input graph passed; no full '
                                                                   'feature qualification'},
                                         'torch': {'status': 'candidate',
                                                   'implementation': 'hcu_adapter.kernels.state_and_constraints.load_state_to_cache',
                                                   'limitations': 'Device-only MTP/state/constraint '
                                                                  'contract; affected upstream refs and '
                                                                  'changed-input graph passed; no full '
                                                                  'feature qualification'}},
 'tilelang_ops.apply_token_bitmask_by_step_inplace_tl': {'triton': {'status': 'candidate',
                                                                    'implementation': 'hcu_adapter.kernels.state_and_constraints.mask_logits',
                                                                    'limitations': 'Device-only '
                                                                                   'MTP/state/constraint '
                                                                                   'contract; affected '
                                                                                   'upstream refs and '
                                                                                   'changed-input graph '
                                                                                   'passed; no full feature '
                                                                                   'qualification'},
                                                         'torch': {'status': 'candidate',
                                                                   'implementation': 'hcu_adapter.kernels.state_and_constraints.mask_logits',
                                                                   'limitations': 'Device-only '
                                                                                  'MTP/state/constraint '
                                                                                  'contract; affected '
                                                                                  'upstream refs and '
                                                                                  'changed-input graph '
                                                                                  'passed; no full feature '
                                                                                  'qualification'}},
 'tilelang_ops.detect_outlier_and_abort_tl': {'triton': {'status': 'candidate',
                                                         'implementation': 'hcu_adapter.kernels.state_and_constraints.detect_outlier',
                                                         'limitations': 'Device-only MTP/state/constraint '
                                                                        'contract; affected upstream refs '
                                                                        'and changed-input graph passed; no '
                                                                        'full feature qualification'},
                                              'torch': {'status': 'candidate',
                                                        'implementation': 'hcu_adapter.kernels.state_and_constraints.detect_outlier',
                                                        'limitations': 'Device-only MTP/state/constraint '
                                                                       'contract; affected upstream refs and '
                                                                       'changed-input graph passed; no full '
                                                                       'feature qualification'}},
 'tilelang_ops.batch_downsample_tl': {'triton': {'status': 'candidate',
                                                 'implementation': 'hcu_adapter.kernels.vision.batch_downsample',
                                                 'limitations': 'Device-only vision/CP metadata contract; '
                                                                'upstream refs and changed-input graph '
                                                                'passed; full vision/CP model qualification '
                                                                'pending'},
                                      'torch': {'status': 'candidate',
                                                'implementation': 'hcu_adapter.kernels.vision.batch_downsample',
                                                'limitations': 'Device-only vision/CP metadata contract; '
                                                               'upstream refs and changed-input graph '
                                                               'passed; full vision/CP model qualification '
                                                               'pending'}},
 'tilelang_ops.prepare_cp_mirror_tl': {'triton': {'status': 'candidate',
                                                  'implementation': 'hcu_adapter.kernels.context_parallel_rccl.prepare_mirror',
                                                  'limitations': 'Device-only vision/CP metadata contract; '
                                                                 'upstream refs and changed-input graph '
                                                                 'passed; full vision/CP model qualification '
                                                                 'pending'},
                                       'torch': {'status': 'candidate',
                                                 'implementation': 'hcu_adapter.kernels.context_parallel_rccl.prepare_mirror',
                                                 'limitations': 'Device-only vision/CP metadata contract; '
                                                                'upstream refs and changed-input graph '
                                                                'passed; full vision/CP model qualification '
                                                                'pending'}},
 'tilelang_ops.replace_input_embeds_tl': {'triton': {'status': 'candidate',
                                                     'implementation': 'hcu_adapter.kernels.vision.replace_input_embeds',
                                                     'limitations': 'Device-only vision/CP metadata '
                                                                    'contract; upstream refs and '
                                                                    'changed-input graph passed; full '
                                                                    'vision/CP model qualification pending'},
                                          'torch': {'status': 'candidate',
                                                    'implementation': 'hcu_adapter.kernels.vision.replace_input_embeds',
                                                    'limitations': 'Device-only vision/CP metadata contract; '
                                                                   'upstream refs and changed-input graph '
                                                                   'passed; full vision/CP model '
                                                                   'qualification pending'}},
 'tilelang_ops.rgb_pad_to_patches_tl': {'triton': {'status': 'candidate',
                                                   'implementation': 'hcu_adapter.kernels.vision.rgb_pad_to_patches',
                                                   'limitations': 'Device-only vision/CP metadata contract; '
                                                                  'upstream refs and changed-input graph '
                                                                  'passed; full vision/CP model '
                                                                  'qualification pending'},
                                        'torch': {'status': 'candidate',
                                                  'implementation': 'hcu_adapter.kernels.vision.rgb_pad_to_patches',
                                                  'limitations': 'Device-only vision/CP metadata contract; '
                                                                 'upstream refs and changed-input graph '
                                                                 'passed; full vision/CP model qualification '
                                                                 'pending'}},
 'tilelang_ops.yoco_cp_boundary_fused_tl': {'triton': {'status': 'candidate',
                                                       'implementation': 'hcu_adapter.kernels.context_parallel_rccl.yoco_boundary',
                                                       'limitations': 'Device-only vision/CP metadata '
                                                                      'contract; upstream refs and '
                                                                      'changed-input graph passed; full '
                                                                      'vision/CP model qualification '
                                                                      'pending'},
                                            'torch': {'status': 'candidate',
                                                      'implementation': 'hcu_adapter.kernels.context_parallel_rccl.yoco_boundary',
                                                      'limitations': 'Device-only vision/CP metadata '
                                                                     'contract; upstream refs and '
                                                                     'changed-input graph passed; full '
                                                                     'vision/CP model qualification '
                                                                     'pending'}}}
