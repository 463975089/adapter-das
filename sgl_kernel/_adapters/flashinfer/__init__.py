"""`flashinfer` (NVIDIA attention/decoding library) adapter for HCU.

sglang imports flashinfer eagerly behind `is_flashinfer_available()` checks in
several attention backends and norm/sampling ops, and the unified fused-op
layer references `sgl_kernel.flash_infer`-style helpers. On HCU flashinfer
cannot be installed (CUDA arch neurons), so this adapter satisfies imports
with loud lazy stubs. Attention on HCU uses the in-tree **triton backend**
(`--attention-backend triton`); flashinfer-backed attention paths fail with a
clear message naming the missing kernel entry.

Per-subtask #229/#234 findings, the main import surfaces covered:
    flashinfer.norm (rmsnorm), flashinfer.cascade.merge_state,
    flashinfer (top_k_top_p_sampling_from_probs, decode/prefill wrappers)
"""

from sgl_kernel._adapters._base import LibStubFinder, is_registered


class _FlashInferFinder(LibStubFinder):
    def __init__(self):
        super().__init__("flashinfer")


def register() -> None:
    if not is_registered(_FlashInferFinder, "flashinfer"):
        import sys

        sys.meta_path.insert(0, _FlashInferFinder())
