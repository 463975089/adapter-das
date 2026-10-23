import os
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
r = dist.get_rank()
torch.cuda.set_device(r)
x = torch.ones(4, device="cuda") * (r + 1)
dist.all_reduce(x)
if r == 0:
    print("ALLREDUCE_OK", x.tolist(), flush=True)
dist.destroy_process_group()
