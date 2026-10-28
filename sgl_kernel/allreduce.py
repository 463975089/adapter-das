"""Tombstone (2026-10-19): the HCU homegrown custom all-reduce is RETIRED.

This module used to re-export the triton custom AR implementation
(``hcu_adapter/triton_ar.py`` + ``ar_ext.cu``: host-shm signaling + HIP IPC
staging). R-01 convicted its cross-block rendezvous (device-scope arrival
atomics do not order the other blocks' host-shm payload stores; skip-think
48/80 vs 1/40 on the aiter route, nmz26 tp4 2026-10-15) and R-13 #1 was
closed as a disposition, NOT a fix: the aiter.dist GraphSafe AR passed the
full stress matrix on the current image (4-rank and 8-rank, 15/15 eager +
graph-replay per topology) and had been the production landing all along,
so the fix was moot and a convicted kernel is useless as a numerics
reference. Files deleted; git history has the protocol archaeology.

Raising ImportError here lands in sglang's designed fallback channel
(``custom_all_reduce_ops`` catches it, flips ``IS_CUSTOM_AR_AVAILABLE``
off and logs this module's message once per boot).
"""

raise ImportError(
    "sgl_kernel.allreduce: HCU custom AR retired 2026-10-19 "
    "(R-01 conviction + R-13#1 disposition; AR tier = aiter.dist "
    "GraphSafe, fallback NCCL. See git history for triton_ar.py."
)
