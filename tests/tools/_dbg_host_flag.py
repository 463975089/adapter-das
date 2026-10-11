"""Host-shm flag visibility probe: GPU0 writes via its device mapping of a
shared /dev/shm page; GPU1 spins reading via its own mapping of the SAME
physical page. If the spin observes the write mid-kernel -> the host-flag
design for the custom AR is viable."""

import mmap
import os
import time

import ctypes
import torch
import torch.distributed as dist

SO = "/root/hcu_t2/sgl_kernel/_adapters/libhcu_ar.so"


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")

    so = ctypes.CDLL(SO)
    so.hcu_register_host.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    so.hcu_register_host.restype = ctypes.c_int
    so.hcu_unregister_host.argtypes = [ctypes.c_void_p]
    so.hcu_flag_write.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p]
    so.hcu_flag_spin.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p]

    path = "/dev/shm/hcu_ar_flag_probe"
    if rank == 0:
        with open(path, "wb") as f:
            f.truncate(4096)
    dist.barrier(group=gloo)
    f = open(path, "r+b")
    mm = mmap.mmap(f.fileno(), 4096)  # MAP_SHARED
    dev_ptr = ctypes.c_void_p()
    rc = so.hcu_register_host(ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm))), 4096, ctypes.byref(dev_ptr))
    assert rc == 0, f"register rc={rc}"
    print(f"[r{rank}] host registered dev_ptr={hex(dev_ptr.value or 0)}", flush=True)
    dist.barrier(group=gloo)

    stream = torch.cuda.current_stream().cuda_stream
    out = torch.zeros(4, dtype=torch.int32, device="cuda")

    for rnd in range(3):
        want = rnd + 1
        if rank == 0:
            dist.barrier(group=gloo)
            time.sleep(0.3)  # let the reader go stale first
            rc = so.hcu_flag_write(dev_ptr.value + rnd * 4, want, stream)
            assert rc == 0
            torch.cuda.synchronize()
            print(f"[r0] wrote flag{rnd}={want}", flush=True)
            dist.barrier(group=gloo)
        else:
            dist.barrier(group=gloo)
            rc = so.hcu_flag_spin(dev_ptr.value + rnd * 4, want, 5_000_000, out.data_ptr(), stream)
            assert rc == 0
            torch.cuda.synchronize()
            d = out.cpu().tolist()
            print(f"[r1] spin flag{rnd}: saw={d[0]} want={want} iters={d[1] << 32 | d[2]}", flush=True)
            dist.barrier(group=gloo)

    dist.barrier(group=gloo)
    if rank == 0:
        print("HOSTFLAG DONE", flush=True)
    dist.destroy_process_group()


main()
