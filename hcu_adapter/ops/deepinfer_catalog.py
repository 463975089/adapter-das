"""Combined catalog for the DeepInfer host bridge."""

from .deepinfer_compute_catalog import BACKEND_CATALOG as COMPUTE_CATALOG
from .deepinfer_aux_catalog import BACKEND_CATALOG as AUX_CATALOG

PLANNED_TILELANG_FACTORIES = ()
RUNTIME_CATALOG = {'runtime.device_visibility': {'ordinal_mask': {'status': 'candidate',
                                                'implementation': 'hcu_adapter.bridges._deepinfer.hcu_env.align_device_visibility',
                                                'limitations': 'Before GPU initialization, align HIP with '
                                                               'upstream numeric physical CUDA assignment '
                                                               'without expanding an explicit HIP mask; '
                                                               'ROCR/UUID mappings unsupported; native DP '
                                                               'topology qualification pending'}},
 'worker.driver_exposure': {'rccl_engram': {'status': 'candidate',
                                            'implementation': 'hcu_adapter.bridges._deepinfer.worker_runtime._prepare_collective_engram_driver',
                                            'limitations': 'RCCL Engram has no NVIDIA peer-memory driver '
                                                           'requirement; defer HIP initialization to Worker; '
                                                           'guard exact active provider classes; original '
                                                           'CP8 offline example completed (attempt61), '
                                                           'formal quality still pending'}},
 'platform.device_inventory': {'drm': {'status': 'candidate',
                                       'implementation': 'hcu_adapter.bridges._deepinfer.platform_guard.install_inventory',
                                       'limitations': 'DRM cardinality through GPU discovery, cache '
                                                      'invalidation and spawn startup; no torch/GPU context; '
                                                      'NUMA mapping remains unavailable, no SM100 identity'}},
 'runtime.upstream_defaults': {'compatible': {'status': 'candidate',
                                              'implementation': 'hcu_adapter.bridges._deepinfer.hcu_env.apply_upstream_defaults',
                                              'limitations': 'Default GPU Engram storage and '
                                                             'non-batch-invariant kernels; eager-only '
                                                             'installer defaults main/draft eager; explicit '
                                                             'environment choices preserved'}},
 'runtime.configuration': {'file': {'status': 'candidate',
                                    'implementation': 'hcu_adapter.bridges._deepinfer.configuration.load',
                                    'limitations': 'Strict JSON/YAML installer options; two configurable '
                                                   'backend groups and two fixed providers; explicit '
                                                   'function arguments win; per-operator hot switching '
                                                   'unsupported'}},
 'runtime.dump': {'json': {'status': 'candidate',
                           'implementation': 'hcu_adapter.bridges._deepinfer.runtime_dump.write',
                           'limitations': 'Atomic per-PID report of actual '
                                          'bindings/catalog/env/version/module paths and installer selection '
                                          'sources; never probes GPU'}},
 'worker.cpu_affinity': {'inherited_mask': {'status': 'candidate',
                                            'implementation': 'hcu_adapter.bridges._deepinfer.worker_runtime.install_cpu_affinity',
                                            'limitations': 'Worker and SO builder share inherited '
                                                           'allowed-CPU partitions; avoids NVIDIA procfs GPU '
                                                           'census; NUMA optimization pending'}},
 'worker.bootstrap.context': {'inherited': {'status': 'candidate',
                                            'implementation': 'hcu_adapter.bridges._deepinfer.bootstrap.publish_worker_context',
                                            'limitations': 'Internal provider context propagated after '
                                                           'explicit complete installation; spawned CP '
                                                           'qualification pending'}},
 'worker.bootstrap.import_hook': {'lazy': {'status': 'candidate',
                                           'implementation': 'hcu_adapter.bridges._deepinfer.bootstrap.register_worker_import_hook',
                                           'limitations': 'No torch import at Python startup; install '
                                                          'providers before the engine is unpickled in CP '
                                                          'children; qualification pending'}},
 'worker.runtime': {'hip_eager': {'status': 'candidate',
                                  'implementation': 'hcu_adapter.bridges._deepinfer.worker_runtime.install',
                                  'limitations': 'Eager by default; explicit allow_cuda_graph=True passed '
                                                 'DP8/EP8/ENP8 CP1 NEXTN0 model smoke (attempt44, 2 rounds); '
                                                 'formal accuracy/long-context pending; PDL/VMM '
                                                 'segments/CuMem RCCL disabled, batch-invariant rejected'}},
 'platform.bf16_tflops_limit': {'guard': {'status': 'candidate',
                                          'implementation': 'hcu_adapter.bridges._deepinfer.platform_guard._bf16_tflops_limit',
                                          'limitations': 'MFU denominator only; BW1101 peak not verified, '
                                                         'defaults to uncalibrated (MFU 0) unless '
                                                         'HCU_BF16_TFLOPS is set'}},
 'vendors.import_surface': {'stub': {'status': 'candidate',
                                     'implementation': 'hcu_adapter.bridges._deepinfer.vendor_imports.install',
                                     'limitations': 'Missing fused MLA, HC GEMM and vision imports only; '
                                                    'calls fail explicitly'}},
 'tile_kernels.import_surface': {'stub': {'status': 'candidate',
                                          'implementation': 'hcu_adapter.bridges._deepinfer.tile_kernels.install',
                                          'limitations': 'Pinned production imports; only separately '
                                                         'registered numerical providers execute'}},
 'deepinfer._C': {'hip_staged': {'status': 'candidate',
                                 'implementation': 'hcu_adapter.bridges._deepinfer.native_extension.install',
                                 'limitations': 'Pinned source build; external GCC13 runtime path; no e2e '
                                                'qualification'}}}

BACKEND_CATALOG = {}
for catalog in (COMPUTE_CATALOG, AUX_CATALOG, RUNTIME_CATALOG):
    duplicates = BACKEND_CATALOG.keys() & catalog.keys()
    if duplicates:
        raise RuntimeError("Duplicate DeepInfer catalog interfaces: " + ", ".join(sorted(duplicates)))
    BACKEND_CATALOG.update(catalog)
