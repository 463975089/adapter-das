# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Read HIP device metadata without initializing a PyTorch device context."""

import ctypes
from functools import lru_cache


@lru_cache(maxsize=1)
def _runtime():
    # Use the runtime already loaded by torch, rather than a second DTK install.
    from pathlib import Path

    for line in Path('/proc/self/maps').read_text().splitlines():
        path = line.split()[-1]
        if path.startswith('/') and any(
            name in path for name in ('/libamdhip64.so', '/libgalaxyhip.so')
        ):
            runtime = ctypes.CDLL(path)
            runtime.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
            runtime.hipGetDevice.restype = ctypes.c_int
            runtime.hipDeviceGetAttribute.argtypes = [
                ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
            ]
            runtime.hipDeviceGetAttribute.restype = ctypes.c_int
            return runtime
    raise RuntimeError('The HIP runtime loaded by torch was not found')


def _check(status, operation):
    if status != 0:
        raise RuntimeError(f'{operation} failed with HIP error {status}')


def device_attribute(attribute, device=None):
    runtime = _runtime()
    if device is None:
        current = ctypes.c_int()
        _check(runtime.hipGetDevice(ctypes.byref(current)), 'hipGetDevice')
        device = current.value
    value = ctypes.c_int()
    _check(
        runtime.hipDeviceGetAttribute(ctypes.byref(value), attribute, device),
        'hipDeviceGetAttribute',
    )
    return value.value


def compute_capability(device=None):
    # hipDeviceAttributeComputeCapabilityMajor/Minor from hipDeviceAttribute_t.
    import torch

    if torch.cuda.is_initialized():
        return torch.cuda.get_device_capability(device)
    device = _device_index(device)
    return device_attribute(36, device), device_attribute(37, device)


def current_device():
    runtime = _runtime()
    value = ctypes.c_int()
    _check(runtime.hipGetDevice(ctypes.byref(value)), 'hipGetDevice')
    return value.value


def _device_index(device):
    import torch

    if device is None:
        return current_device()
    if isinstance(device, int):
        return device
    parsed = torch.device(device)
    if parsed.type != 'cuda':
        raise ValueError(f'Expected a CUDA device, got {device}')
    return current_device() if parsed.index is None else parsed.index


@lru_cache(maxsize=None)
def _architecture_for_index(device):
    import amdsmi

    runtime = _runtime()
    query = runtime.hipDeviceGetPCIBusId
    query.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    bdf = ctypes.create_string_buffer(32)
    _check(query(bdf, len(bdf), device), "hipDeviceGetPCIBusId")
    address = bdf.value.decode().lower()
    amdsmi.amdsmi_init()
    for handle in amdsmi.amdsmi_get_processor_handles():
        if amdsmi.amdsmi_get_gpu_device_bdf(handle).lower() == address:
            arch = amdsmi.amdsmi_get_gpu_asic_info(handle)["target_graphics_version"]
            if not arch.startswith("gfx"):
                raise RuntimeError(f"Invalid SMI architecture for {address}: {arch!r}")
            return arch
    raise RuntimeError(f"No SMI device matches HIP PCI address {address}")


def architecture(device=0):
    """Resolve the visible HIP ordinal by PCI address, not SMI enumeration order."""
    return _architecture_for_index(_device_index(device))


def multiprocessor_count(device=None):
    return device_attribute(32, _device_index(device))


class _Properties:
    """Only metadata needed by the audited import-time consumers.

    Once a worker has initialized torch, use its canonical properties object.
    Unknown early queries fail rather than silently initializing GPU0.
    """

    def __init__(self, device):
        self.device = device

    def __getattr__(self, name):
        import torch

        if torch.cuda.is_initialized():
            return getattr(torch.cuda.get_device_properties(self.device), name)
        if name == 'gcnArchName':
            # Audited early consumers use the gfx family/prefix only. Once
            # initialized, the canonical property (including suffixes) is used.
            return architecture(self.device)
        attributes = {'major': 36, 'minor': 37, 'multi_processor_count': 32,
                      'warp_size': 27}
        if name in attributes:
            return device_attribute(attributes[name], self.device)
        if name == 'name':
            query = _runtime().hipDeviceGetName
            query.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            query.restype = ctypes.c_int
            value = ctypes.create_string_buffer(256)
            _check(query(value, len(value), self.device), 'hipDeviceGetName')
            return value.value.decode()
        raise AttributeError(f'Unsupported pre-context HIP metadata field: {name}')


def device_properties(device=None):
    import torch

    if torch.cuda.is_initialized():
        return torch.cuda.get_device_properties(device)
    return _Properties(_device_index(device))


def total_memory(device=None):
    query = _runtime().hipDeviceTotalMem
    query.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    query.restype = ctypes.c_int
    value = ctypes.c_size_t()
    _check(query(ctypes.byref(value), _device_index(device)), 'hipDeviceTotalMem')
    return value.value


def minimum_memory_capacity():
    import torch

    count = torch.cuda.device_count()
    if not count:
        raise RuntimeError('No visible HIP devices for memory capacity query')
    return min(total_memory(device) // (1024 * 1024) for device in range(count))
