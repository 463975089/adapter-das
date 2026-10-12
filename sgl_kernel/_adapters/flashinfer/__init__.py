"""`flashinfer` (NVIDIA attention/decoding library) adapter for HCU.

sglang imports flashinfer eagerly behind `is_flashinfer_available()` checks in
several attention backends and norm/sampling ops, and the unified fused-op
layer references `sgl_kernel.flash_infer`-style helpers. On HCU flashinfer
cannot be installed (CUDA arch neurons), so this adapter satisfies imports
with loud lazy stubs. Attention on HCU runs the in-tree attention backends
(fa3/triton); flashinfer-backed attention paths fail with a clear message
naming the missing kernel entry.

Per-subtask #229/#234 findings, the main import surfaces covered:
    flashinfer.norm (rmsnorm), flashinfer.cascade.merge_state,
    flashinfer (top_k_top_p_sampling_from_probs, decode/prefill wrappers)

§0.5 is_cuda 路线追加（2026-10-12, bw7）：is_cuda 语义让上游走进 CUDA 分支
的 flashinfer 特性探测，探测必须回答"没有"而不是"有但坏"：
    - flashinfer.comm.mnnvl          -> deny（上游两处 except ImportError
                                        兜底，回默认 process group）
    - comm.allreduce_fusion 族       -> ABSENT（上游 hasattr 探测失败走
                                        standard-allreduce 回退 + unavailable
                                        旗标，与 HCU 自家 AR 路线一致）
"""

from sgl_kernel._adapters._base import ABSENT, LibStubFinder, is_registered


class _FlashInferFinder(LibStubFinder):
    def __init__(self):
        super().__init__(
            "flashinfer",
            allowlist={
                "flashinfer.comm": {
                    "allreduce_fusion": ABSENT,
                    "create_allreduce_fusion_workspace": ABSENT,
                },
            },
            deny=("flashinfer.comm.mnnvl",),
        )


def register() -> None:
    if not is_registered(_FlashInferFinder, "flashinfer"):
        import sys

        sys.meta_path.insert(0, _FlashInferFinder())

    # sampler face: sglang's is_cuda branch imports from flashinfer.sampling;
    # bind to the same torch-reference kernels sgl_kernel exports (§0.5).
    try:
        from sgl_kernel.sampling import (
            min_p_sampling_from_probs,
            top_k_top_p_sampling_from_probs,
        )

        _FI_SAMPLING_ALLOW = {
            "flashinfer.sampling": {
                "min_p_sampling_from_probs": min_p_sampling_from_probs,
                "top_k_top_p_sampling_from_probs": top_k_top_p_sampling_from_probs,
            },
        }
        _f = next(
            f
            for f in sys.meta_path
            if isinstance(f, _FlashInferFinder)
        )
        _f.allowlist.update(_FI_SAMPLING_ALLOW)
        try:
            from sgl_kernel._adapters import registry as _reg

            _reg.register(
                "sampling",
                min_p_sampling_from_probs,
                target="flashinfer.sampling:min_p/topk_topp",
                mechanism=_reg.MECH_SEAM,
                detail="torch reference tier; sglang sampler face",
            )
        except Exception:
            pass
    except Exception:
        pass
