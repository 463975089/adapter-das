# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Zero-workspace, runtime-qualified vendor FP8 GEMM; unsupported stays native."""
from __future__ import annotations
import ctypes
from functools import lru_cache
import hashlib
import logging
from pathlib import Path
import subprocess
import tempfile
import torch

@lru_cache(maxsize=1)
def _library():
    # The installed headers compile the ABI; never deserialize an opaque algo
    # from a different vendor library. Compilation happens only outside capture.
    import fcntl
    source=Path(__file__).with_suffix('.cpp')
    root=Path('/opt/dtk')
    header=root/'include/hipblaslt/hipblaslt.h'
    library=root/'lib/libhipblaslt.so'
    if not source.is_file() or not header.is_file() or not library.is_file():return None
    signature=hashlib.sha256(source.read_bytes()+header.read_bytes()+
        str(library.resolve().stat()).encode()).hexdigest()[:24]
    directory=Path.home()/'.cache/hcu_adapter/fp8_lt'/signature
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        target=directory/'kernel.so'
        if not target.exists():
            try:
                with tempfile.TemporaryDirectory(dir=directory) as temporary:
                    built=Path(temporary)/'kernel.so'
                    command=['g++','-std=c++17','-D__HIP_PLATFORM_AMD__','-fmax-errors=3',
                        '-O2','-shared','-fPIC','-I'+str(root/'include'),'-L'+str(root/'lib'),
                        '-Wl,-rpath,'+str(root/'lib'),str(source),'-lhipblaslt','-o',str(built)]
                    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
                    (directory/'build.log').write_bytes(result.stdout)
                    if result.returncode:raise RuntimeError('vendor extension compile failed')
                    built.replace(target)
            except (OSError,subprocess.SubprocessError,RuntimeError) as error:
                logging.getLogger('hcu.adapter').info('FP8 vendor heuristic unavailable: %s',error)
                return None
    lib=ctypes.CDLL(str(target));fn=lib.hcu_fp8_lt_run
    fn.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*6+[ctypes.c_int]
    fn.restype=ctypes.c_int
    return lib

@lru_cache(maxsize=16)
def _device_supported(device):
    properties=torch.cuda.get_device_properties(device)
    return (str(getattr(properties,'gcnArchName','')).split(':')[0]=='gfx938'
            and properties.multi_processor_count==64)

def try_scaled_mm(a,w,sa,sw):
    """a[M,K], w[N,K], FP32 token/channel scales; output BF16 or None.

    Shape families are bounded by the vendor tile domain, never by one batch
    size or request length. Unseen capture shapes retain the original provider.
    """
    if (not a.is_cuda or a.dtype!=torch.float8_e4m3fn or w.dtype!=a.dtype
        or a.ndim!=2 or w.ndim!=2 or not a.is_contiguous() or not w.is_contiguous()
        or w.device!=a.device or not _device_supported(a.device)):return None
    m,k=a.shape;n=w.shape[0]
    if not (0<m<=256 and n>=1024 and n%16==0 and k>=1024
            and k%128==0 and w.shape[1]==k):return None
    if (sa.dtype!=torch.float32 or sw.dtype!=torch.float32 or sa.device!=a.device
        or sw.device!=a.device or not sa.is_contiguous() or not sw.is_contiguous()
        or sa.numel()!=m or sw.numel()!=n):return None
    # The C ABI creates handles and keys its cache with hipGetDevice. Capture
    # state also belongs to the current device's stream, not the caller's GPU.
    # Restore the caller even when the provider falls back or raises.
    with torch.cuda.device(a.device):
        capturing=torch.cuda.is_current_stream_capturing()
        if capturing and _library.cache_info().currsize==0:return None
        lib=_library()
        if lib is None:return None
        out=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
        result=lib.hcu_fp8_lt_run(m,n,k,a.data_ptr(),w.data_ptr(),sa.data_ptr(),sw.data_ptr(),
            out.data_ptr(),torch.cuda.current_stream(a.device).cuda_stream,not capturing)
        if result<0:raise RuntimeError('FP8 vendor heuristic execution failed')
        return out if result else None
