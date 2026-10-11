"""Ground-truth dump after ONE eager AR call (fp32, n=1024):
  - my staged slot (hipMemcpy D2H from own buffer)
  - peer's slot (hipMemcpy D2H through the IPC mapping, host-issued)
  - the AR output (what the kernel computed)
  - a post-hoc single-thread kernel read of the peer slot (hcu_stage_read)
Discriminates: staging never landed / landed but in-AR read missed it /
even post-hoc reads miss it."""

import ctypes
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, "/root/hcu_t2")
from sgl_kernel._adapters import triton_ar  # noqa: E402

SO = "/root/hcu_t2/sgl_kernel/_adapters/libhcu_ar.so"


def d2h(hip, addr, nbytes):
    buf = (ctypes.c_char * nbytes)()
    rc = hip.hipMemcpy(buf, ctypes.c_void_p(addr), nbytes, 2)
    assert rc == 0, rc
    return bytes(buf)


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")

    so = ctypes.CDLL(SO)
    so.hcu_stage_read.argtypes = [ctypes.c_longlong, ctypes.c_longlong,
                                  ctypes.c_longlong, ctypes.c_longlong,
                                  ctypes.c_longlong, ctypes.c_void_p]
    so.hcu_stage_read.restype = ctypes.c_int

    hb = triton_ar.allocate_meta_buffer(triton_ar.DATA_OFF + 1024 * 1024)
    handles = [None] * 2
    dist.all_gather_object(handles, hb, group=gloo)
    st = triton_ar.init_custom_ar(None, None, handles, None, rank, True)
    print(f"[r{rank}] init ok slots={st.slots} max={st._max_size}", flush=True)

    n = 1024
    x = torch.full((n,), float(rank + 1), dtype=torch.float32, device="cuda")
    out = st.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()

    words = st.read_sync_words()
    seq = words[0]
    slot = seq % st.slots  # committed seq == the round k just executed
    print(f"[r{rank}] seq={seq} slot={slot} out[:4]={out[:4].tolist()} "
          f"out.sum={out.sum().item()}", flush=True)

    off = triton_ar.DATA_OFF + slot * st._max_size
    mine = d2h(triton_ar._HIP, st._own_base + off, 16)
    print(f"[r{rank}] my slot first16B: {mine.hex()}", flush=True)
    peer = d2h(triton_ar._HIP, st._peer_data[0] + slot * st._max_size, 16)
    print(f"[r{rank}] peer slot (host memcpy) first16B: {peer.hex()}", flush=True)

    # post-hoc kernel read of the peer slot (flag spins on a dummy always-set
    # word: pass budget=0 so it skips the wait entirely)
    res = torch.zeros(8, dtype=torch.float32, device="cuda")
    rc = so.hcu_stage_read(st._peer_data[0] + slot * st._max_size, 4,
                           st._flags, 0, res.data_ptr(),
                           torch.cuda.current_stream().cuda_stream)
    assert rc == 0, rc
    torch.cuda.synchronize()
    print(f"[r{rank}] post-hoc kernel read peer[0:4]: {res[1:5].tolist()}",
          flush=True)

    triton_ar.dispose(st)
    dist.barrier()
    dist.destroy_process_group()


main()
