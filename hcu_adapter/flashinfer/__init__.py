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
    - flashinfer.gdn_prefill/gdn_decode/gdn_kernels
                                     -> deny（上游 _get_flashinfer_gdn_kernels
                                        except ImportError 兜底答 available=
                                        False → flashinfer_gdn_prefill_default
                                        回 None → GDN prefill 回 triton；
                                        2026-10-13：Qwen3.5-27B e2e 首验暴露，
                                        此前响亮 stub 让 loader 内 except 只接
                                        ImportError 不接 NotImplementedError，
                                        未皂数）
"""

from hcu_adapter._base import ABSENT, LibStubFinder, is_registered


class _FlashInferFinder(LibStubFinder):
    def __init__(self):
        super().__init__(
            "flashinfer",
            allowlist={
                "flashinfer.comm": {
                    "allreduce_fusion": ABSENT,
                    "create_allreduce_fusion_workspace": ABSENT,
                },
                # raise AttributeError on getattr so `from flashinfer.decode
                # import X` falls back to a submodule import, which the deny
                # entry below then refuses -> from-import raises ImportError
                # (caught by flashinfer_backend.py:2516 -> GQA heuristic).
                # Either half alone fails: plain-missing binds a Missing
                # handle (loud NotImplementedError leaks through the
                # caller's except), ABSENT-without-deny stubs X as a
                # submodule ("module object is not callable").
                "flashinfer.decode": {
                    "_grouped_size_compiled_for_decode_kernels": ABSENT,
                },
            },
            deny=(
                "flashinfer.comm.mnnvl",
                "flashinfer.gdn_prefill",
                "flashinfer.gdn_decode",
                "flashinfer.gdn_kernels",
                "flashinfer.decode._grouped_size_compiled_for_decode_kernels",
            ),
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
            softmax,
            top_k_top_p_sampling_from_probs,
        )

        _FI_SAMPLING_ALLOW = {
            "flashinfer.sampling": {
                "min_p_sampling_from_probs": min_p_sampling_from_probs,
                "softmax": softmax,
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
            from hcu_adapter import registry as _reg

            _reg.register(
                "sampling",
                min_p_sampling_from_probs,
                target="flashinfer.sampling:min_p/topk_topp",
                mechanism=_reg.MECH_SEAM,
                detail="torch reference tier; sglang sampler face",
            )
            _reg.register(
                "sampling",
                softmax,
                target="flashinfer.sampling:softmax",
                mechanism=_reg.MECH_SEAM,
                detail="triton port (online 2-pass row softmax with per-row"
                " temperature); dspark verify SoftmaxTemp hot path",
            )
        except Exception:
            pass
    except Exception:
        pass
