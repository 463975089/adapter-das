# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_adapter import hcu_env

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
        return None
    return _PLATFORM_CLASS


# anchor-baseline(④): main@5792cf6e diffusion TP passes use_srt_custom_allreduce directly to GroupCoordinator.
def install_collective_policy():
    from hcu_adapter import kill_switch, registry

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


def get_platform_class():
    global _PLATFORM_TYPE
    if _PLATFORM_TYPE is not None:
        return _PLATFORM_TYPE

    from sglang.multimodal_gen.runtime.platforms import PlatformEnum
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
        def optimize_vae(cls, vae):
            return vae

        def init_backend(self):
            from hcu_adapter import registry

            registry.register(
                'platform_detect', 'cuda',
                target='sglang.multimodal_gen.runtime.platforms.current_platform',
                mechanism=registry.MECH_SEAM,
                detail='HCU diffusion OOT; native CUDA semantics, HIP hardware and non-NVML metadata',
            )
            install_collective_policy()

    _PLATFORM_TYPE = HcuDiffusionPlatform
    return HcuDiffusionPlatform
