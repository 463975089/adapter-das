import ctypes
import logging
from typing import Tuple
import torch

logger = logging.getLogger(__name__)
_HIP_RUNTIME = None

def _hip_runtime():
    global _HIP_RUNTIME
    if _HIP_RUNTIME is False:
        return None
    if _HIP_RUNTIME is not None:
        return _HIP_RUNTIME
    for library in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            lib = ctypes.CDLL(library)
        except OSError:
            continue
        lib.hipHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        lib.hipHostRegister.restype = ctypes.c_int
        lib.hipHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        lib.hipHostGetDevicePointer.restype = ctypes.c_int
        _HIP_RUNTIME = lib
        return lib
    _HIP_RUNTIME = False
    return None


def _device_ptr_for_host_tensor(tensor: torch.Tensor) -> int:
    """HCU kernels cannot dereference the raw host pointer; translate it."""
    host_ptr = int(tensor.data_ptr())
    hip = _hip_runtime()
    if hip is None:
        return host_ptr
    device_ptr = ctypes.c_void_p()
    rc = int(
        hip.hipHostGetDevicePointer(
            ctypes.byref(device_ptr), ctypes.c_void_p(host_ptr), ctypes.c_uint(0)
        )
    )
    if rc == 0 and device_ptr.value:
        return int(device_ptr.value)
    logger.warning(
        "PLE hipHostGetDevicePointer failed rc=%s host=%#x; using host pointer",
        rc,
        host_ptr,
    )
    return host_ptr


class _CudaArrayInterface:
    """Non-owning CUDA array interface over a mapped host allocation."""

    def __init__(self, ptr: int, shape: Tuple[int, ...], typestr: str) -> None:
        self.__cuda_array_interface__ = {
            "data": (int(ptr), False),
            "shape": tuple(shape),
            "typestr": typestr,
            "version": 3,
        }


def _wrap_mapped_host_as_cuda_tensor(
    ptr: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, _CudaArrayInterface]:
    """Expose mapped GTT as a CUDA tensor so ATen gather is HIP-graph-safe.

    Triton ``tl.load`` from a host-mapped pointer is not replayed correctly
    inside HIP full CUDA graphs (decode collapses to token 0). ``F.embedding``
    on this view is captured and replays the current ids.
    """
    if dtype == torch.bfloat16:
        typestr = "|u2"
    elif dtype == torch.int8:
        typestr = "|i1"
    elif dtype == torch.float8_e4m3fn:
        typestr = "|u1"
    else:
        raise TypeError(f"unsupported PLE mapped-view dtype: {dtype}")
    holder = _CudaArrayInterface(ptr, shape, typestr)
    tensor = torch.as_tensor(holder, device=device)
    if dtype == torch.bfloat16:
        tensor = tensor.view(torch.bfloat16)
    elif dtype == torch.float8_e4m3fn:
        tensor = tensor.view(torch.float8_e4m3fn)
    from .registry import register, MECH_SEAM
    register("mapped_host_view", "dtk", __name__+"._wrap_mapped_host_as_cuda_tensor", MECH_SEAM, source=__name__)
    return tensor, holder
