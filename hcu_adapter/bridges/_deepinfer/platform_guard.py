"""SM100-only platform answers that HCU must replace (governance §0.5 guardrail list).

deep_cluster.hardware.get_bf16_tflops_limit raises for anything but SM100 /
Ascend; deepinfer's LLMEngine only uses it as the MFU denominator. BW1101 has
no verified peak in this repo, so the default is "uncalibrated" (infinite peak,
MFU reported as 0) unless HCU_BF16_TFLOPS supplies the device spec value.
"""

from . import hcu_env
from hcu_adapter import registry


def install_inventory(physical_device_count):
    """Expose DRM cardinality through GPU discovery without creating a context."""
    from pathlib import Path
    import deep_cluster.hardware.device as device
    import deep_cluster.hardware.nvidia as nvidia

    if not hasattr(nvidia, '_hcu_original_physical_device_count'):
        nvidia._hcu_original_physical_device_count = nvidia.physical_device_count
        nvidia._hcu_original_is_available = nvidia.is_available

    def count(*, _root=Path('/')):
        paths = list((_root / 'sys/class/drm').glob('card[0-9]*/device/mem_info_vram_total'))
        if _root == Path('/'):
            if len(paths) != physical_device_count:
                raise RuntimeError(f'HCU inventory has {len(paths)} devices; installer expected {physical_device_count}')
            return len(paths)
        return len(paths) if paths else nvidia._hcu_original_physical_device_count(_root=_root)

    def available(*, _root=Path('/')):
        return bool(count(_root=_root)) or nvidia._hcu_original_is_available(_root=_root)

    nvidia.physical_device_count = count
    nvidia.is_available = available
    for function in (device.accelerator_kind, device.physical_device_count, device.visible_device_env_name,
                     device.device_numa_node, device.is_sm_major_10):
        function.cache_clear()
    if device.accelerator_kind() != 'nvidia':
        raise RuntimeError('The HCU adapter requires GPU semantics; mixed accelerator discovery is unsupported')
    registry.register('platform.device_inventory', 'hcu_adapter.bridges._deepinfer.platform_guard.install_inventory',
                      'deep_cluster.hardware.nvidia.is_available/physical_device_count', 'rebind')


def _bf16_tflops_limit():
    value = hcu_env.resolve("HCU_BF16_TFLOPS")
    return value if value > 0 else float("inf")


def install():
    import deep_cluster.hardware
    import deep_cluster.hardware.device as device

    # llm_engine imports the symbol by name, so rebind every holder before it is imported.
    for module in (device, deep_cluster.hardware):
        module.get_bf16_tflops_limit = _bf16_tflops_limit
    import sys

    engine = sys.modules.get("deepinfer.engine.llm_engine")
    if engine is not None:
        engine.get_bf16_tflops_limit = _bf16_tflops_limit
    registry.register("platform.bf16_tflops_limit", "hcu_adapter.bridges._deepinfer.platform_guard._bf16_tflops_limit",
                      "deep_cluster.hardware.get_bf16_tflops_limit", "rebind")
    source = "HCU_BF16_TFLOPS" if hcu_env.resolve("HCU_BF16_TFLOPS") > 0 else "uncalibrated (MFU reported as 0)"
    print("HCU adapter runtime: bf16 TFLOPS limit -> " + source, flush=True)
