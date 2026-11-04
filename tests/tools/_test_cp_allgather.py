# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""2-rank NCCL all_gather_into_tensor sanity on HCU (torch.distributed).

Mimics zigzag._all_gather_reorganized shapes: rank0 sends 4 rows, rank1
sends 2 rows (padded to 4); verify each rank's gathered tensor matches.
Run: torchrun --nproc_per_node=2 with HIP_VISIBLE_DEVICES=2,3.
"""
import os

import torch
import torch.distributed as dist

RANK = int(os.environ.get("RANK", "0"))


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dev = f"cuda:{os.environ.get('LOCAL_RANK', '0')}"

    for rows, dim in ((4, 512), (2, 512), (1194, 4096), (1, 4096)):
        max_len = 4 if rows <= 4 else rows
        local = torch.randn(rows, dim, device=dev, dtype=torch.bfloat16)
        if rows < max_len:
            local = torch.cat(
                [local, torch.zeros(max_len - rows, dim, device=dev, dtype=torch.bfloat16)]
            )
        gathered = torch.empty(max_len * 2, dim, device=dev, dtype=torch.bfloat16)
        dist.all_gather_into_tensor(gathered, local.contiguous())
        # each rank checks the OTHER rank's slice via its own ground truth:
        # build expected from a synchronized copy
        ref = torch.empty_like(gathered)
        dist.all_gather_into_tensor(ref, local)
        ok_vals = torch.allclose(gathered, ref)
        # cross-check content: rank r fills its slice with r*1000+arange
        stamp = torch.full((max_len, dim), float(RANK * 1000), device=dev, dtype=torch.bfloat16)
        stamp += torch.arange(max_len, device=dev, dtype=torch.bfloat16).unsqueeze(1)
        g2 = torch.empty_like(stamp).repeat(2, 1)
        dist.all_gather_into_tensor(g2, stamp.contiguous())
        expect0 = torch.full((max_len, dim), 0.0, device=dev, dtype=torch.bfloat16) + torch.arange(
            max_len, device=dev, dtype=torch.bfloat16
        ).unsqueeze(1)
        expect1 = expect0 + 1000
        ok_stamp = torch.allclose(g2[:max_len], expect0) and torch.allclose(
            g2[max_len:], expect1
        )
        print(
            f"rank{RANK} rows={rows}: vals_ok={ok_vals} stamp_ok={ok_stamp}",
            flush=True,
        )
        if not (ok_vals and ok_stamp):
            print(f"rank{RANK} FAIL g2[:2,0]", g2[:2, 0].float().tolist(), flush=True)
            dist.destroy_process_group()
            return 1
    if RANK == 0:
        print("ALL OK", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
