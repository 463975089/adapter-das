"""Data-transfer probe: writer GPU stages a pattern into ITS OWN device
buffer (plain stores + fence variant), publishes via host flag; a CONCURRENT
reader kernel on the other GPU spins the host flag then plain-reads the
writer's buffer through the IPC mapping. Isolates whether C++ staged device
data is visible to a concurrent remote reader."""

import ctypes
import mmap
import os
import time

import torch
import torch.distributed as dist

SO = "/root/hcu_t2/sgl_kernel/_adapters/libhcu_ar.so"


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")

    so = ctypes.CDLL(SO)
    so.hcu_register_host.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.POINTER(ctypes.c_void_p)]
    so.hcu_register_host.restype = ctypes.c_int
    so.hcu_stage_write.argtypes = [ctypes.c_longlong, ctypes.c_longlong,
                                   ctypes.c_float, ctypes.c_longlong,
                                   ctypes.c_int, ctypes.c_void_p]
    so.hcu_stage_write.restype = ctypes.c_int
    so.hcu_stage_read.argtypes = [ctypes.c_longlong, ctypes.c_longlong,
                                  ctypes.c_longlong, ctypes.c_longlong,
                                  ctypes.c_longlong, ctypes.c_void_p]
    so.hcu_stage_read.restype = ctypes.c_int

    import sys
    sys.path.insert(0, "/root/hcu_t2")
    from sgl_kernel._adapters import triton_ar

    hb = triton_ar.allocate_meta_buffer(triton_ar.DATA_OFF + 1024 * 1024)
    handles = [None] * 2
    dist.all_gather_object(handles, hb, group=gloo)

    my_base = triton_ar._ACTIVE_ALLOCATIONS[bytes(hb)][0]
    peer_base = None
    if rank != 0:
        hd = triton_ar.IpcMemHandle.from_buffer_copy(handles[0])
        addr = ctypes.c_void_p()
        rc = triton_ar._HIP.hipIpcOpenMemHandle(
            ctypes.byref(addr), hd, triton_ar.HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
        )
        assert rc == 0, rc
        peer_base = addr.value

    # shared host flag page
    path = "/dev/shm/hcu_ar_data_probe"
    if rank == 0:
        with open(path, "wb"):
            pass
        with open(path, "r+b") as f:
            f.truncate(4096)
    dist.barrier(group=gloo)
    f = open(path, "r+b")
    mm = mmap.mmap(f.fileno(), 4096)
    mm[0:8] = b"\x00" * 8
    dev_ptr = ctypes.c_void_p()
    rc = so.hcu_register_host(
        ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm))), 4096,
        ctypes.byref(dev_ptr))
    assert rc == 0, rc
    dist.barrier(group=gloo)

    stream = torch.cuda.current_stream().cuda_stream
    out = torch.zeros(8, dtype=torch.float32, device="cuda")
    n = 1024
    data_off = triton_ar.DATA_OFF

    for mode in (0, 1):
        for rnd in range(3):
            val = 100.0 + mode * 10 + rnd
            if rank == 0:
                dist.barrier(group=gloo)
                time.sleep(0.2)  # let the reader park on the flag first
                rc = so.hcu_stage_write(my_base + data_off, n, val,
                                        dev_ptr.value, mode, stream)
                assert rc == 0, rc
                torch.cuda.synchronize()
                dist.barrier(group=gloo)
            else:
                dist.barrier(group=gloo)
                rc = so.hcu_stage_read(peer_base + data_off, n, dev_ptr.value,
                                       20_000_000, out.data_ptr(), stream)
                assert rc == 0, rc
                torch.cuda.synchronize()
                d = out.cpu().tolist()
                ok = d[2] == val and d[1] == val
                print(f"[r1] mode={mode} rnd={rnd}: flag={d[0]:.0f} "
                      f"min={d[1]} max={d[2]} first={d[3]} "
                      f"{'PASS' if ok else 'FAIL'} (want {val})", flush=True)
                # reset flag for next round via CPU
                mm[0:8] = b"\x00" * 8
                dist.barrier(group=gloo)

    dist.barrier(group=gloo)
    if rank == 0:
        print("DATAPROBE DONE", flush=True)
    dist.destroy_process_group()


main()
