# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_adapter.infra import hcu_env

_PLATFORM_CLASS = 'hcu_adapter.bridges.sglang.HcuDiffusionPlatform'
_PLATFORM_TYPE = None
_SUPPORTED_ARCHES = frozenset({'gfx936'})


# anchor-baseline(②): main@5792cf6e provides diffusion OOT platforms and NonNvmlCudaPlatform.
def platform_plugin():
    if hcu_env.resolve('HCU_PLATFORM_SEMANTICS')[0] != 'cuda':
        return None
    import torch

    if torch.version.hip is None or torch.cuda.device_count() == 0:
        return None
    architecture = hcu_env.arch_name()
    if not architecture:
        raise RuntimeError('Cannot resolve the visible HIP architecture for diffusion')
    if architecture not in _SUPPORTED_ARCHES:
        from hcu_adapter.bridges.sglang_diffusion_numerics import warn_unsupported_arches

        warn_unsupported_arches((architecture,))
        return None
    return _PLATFORM_CLASS


# anchor-baseline(④): main@5792cf6e diffusion TP passes use_srt_custom_allreduce directly to GroupCoordinator.
def install_collective_policy():
    from hcu_adapter.infra import registry
    from hcu_adapter.infra import kill_switch

    refused = kill_switch.op_disabled('aiter_ar')
    if not refused:
        value, _ = hcu_env.resolve('HCU_AITER_AR')
        refused = str(value) in ('', '0', 'None') or str(value).startswith('refused')
    if not refused:
        return False
    import functools
    import inspect
    from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator

    original = GroupCoordinator.__init__
    if getattr(original, '_hcu_diffusion_custom_ar_off', False):
        return False
    signature = inspect.signature(original)
    if 'use_srt_custom_allreduce' not in signature.parameters:
        raise RuntimeError('Diffusion GroupCoordinator custom allreduce contract changed')

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        bound.arguments['use_srt_custom_allreduce'] = False
        return original(*bound.args, **bound.kwargs)

    initialize._hcu_diffusion_custom_ar_off = True
    with registry.registration_transaction():
        registry.register(
            'allreduce', 'nccl',
            target='sglang.multimodal_gen.runtime.distributed.group_coordinator.GroupCoordinator.__init__',
            mechanism=registry.MECH_REBIND,
            detail='Existing HCU custom-AR off/refusal policy; native Torch RCCL fallback',
        )
        GroupCoordinator.__init__ = initialize
    return True


class _HcuFa2Unavailable(ValueError):
    pass


def _select_hcu_fa2(head_size, dtype, *, explicit_fa2):
    from importlib.machinery import PathFinder

    import torch

    from hcu_adapter.infra import kill_switch, registry

    if kill_switch.op_disabled('vendor_fa'):
        raise _HcuFa2Unavailable('HCU diffusion FA2 is disabled by vendor_fa')
    if dtype not in (torch.bfloat16, torch.float16):
        raise _HcuFa2Unavailable('HCU diffusion FA2 requires BF16 or FP16')
    if PathFinder.find_spec('flash_attn') is None:
        raise _HcuFa2Unavailable('HCU diffusion FA2 requires the DTK flash_attn wheel')
    try:
        from hcu_adapter.bridges.sglang_diffusion_fa2 import (
            HcuFlashAttention2Backend,
            HcuFlashAttentionBackend,
        )
    except (ImportError, OSError) as error:
        raise _HcuFa2Unavailable(
            'HCU diffusion FA2 could not load the DTK flash_attn interfaces'
        ) from error
    backend = HcuFlashAttention2Backend if explicit_fa2 else HcuFlashAttentionBackend
    if head_size not in backend.get_supported_head_sizes():
        raise _HcuFa2Unavailable(f'HCU diffusion FA2 does not support head size {head_size}')

    registry.register(
        'diffusion_fa2',
        backend,
        target='sglang.multimodal_gen.runtime.platforms.current_platform.get_attn_backend_cls_str',
        mechanism=registry.MECH_SEAM,
        detail='DTK FA2 packed varlen with requested backend identity',
    )
    return f'{backend.__module__}.{backend.__qualname__}'


def get_platform_class():
    global _PLATFORM_TYPE
    if _PLATFORM_TYPE is not None:
        return _PLATFORM_TYPE

    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum, PlatformEnum
    from sglang.multimodal_gen.runtime.platforms.cuda import NonNvmlCudaPlatform
    from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform

    class HcuDiffusionPlatform(NonNvmlCudaPlatform):
        _enum = PlatformEnum.OOT
        device_name = 'hcu'

        @classmethod
        def is_cuda_static(cls):
            return True

        def is_cuda_alike(self):
            return True

        def get_dispatch_key_name(self):
            return 'cuda'

        def get_device(self, local_rank):
            import torch

            return torch.device(self.device_type, local_rank)

        def supports_distributed_device_id(self):
            return True

        @classmethod
        def device_shares_host_memory(cls):
            return False

        @classmethod
        def get_available_gpu_memory(
            cls, device_id=None, distributed=False, empty_cache=True, cpu_group=None
        ):
            return RocmPlatform.get_available_gpu_memory.__func__(
                cls, device_id=device_id, distributed=distributed,
                empty_cache=empty_cache, cpu_group=cpu_group,
            )

        @classmethod
        def get_attn_backend_cls_str(cls, selected_backend, head_size, dtype):
            if selected_backend in (AttentionBackendEnum.FA, AttentionBackendEnum.FA2):
                try:
                    return _select_hcu_fa2(
                        head_size,
                        dtype,
                        explicit_fa2=selected_backend is AttentionBackendEnum.FA2,
                    )
                except _HcuFa2Unavailable:
                    if selected_backend is AttentionBackendEnum.FA2:
                        raise
            return super().get_attn_backend_cls_str(selected_backend, head_size, dtype)

        @classmethod
        def optimize_vae(cls, vae):
            return vae

        def init_backend(self):
            from hcu_adapter.bridges import sglang_diffusion_numerics
            from hcu_adapter.infra import registry

            registry.register(
                'platform_detect', 'cuda',
                target='sglang.multimodal_gen.runtime.platforms.current_platform',
                mechanism=registry.MECH_SEAM,
                detail='HCU diffusion OOT; native CUDA semantics, HIP hardware and non-NVML metadata',
            )
            sglang_diffusion_numerics.install()
            install_collective_policy()

    _PLATFORM_TYPE = HcuDiffusionPlatform
    return HcuDiffusionPlatform
