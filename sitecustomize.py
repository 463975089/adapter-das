"""HCU compatibility bridges for the upstream sglang stack.

Loaded automatically (as `sitecustomize`) whenever PYTHONPATH includes this
directory, including spawn'd sglang scheduler children. sglang and vllm files
stay untouched; what we bridge here:

1. vllm._custom_ops.rms_norm / fused_add_rms_norm -> sgl_kernel triton kernels.
   The HIP vllm build's `torch.ops._C` extension does not carry these ops, so
   sglang's layernorm forward_hip would crash at call time (the import-time
   `_has_vllm_rms_norm` check passes). We replace the python wrappers with
   bridges to our kernel — same signatures, fp32-accumulated math.
2. `aiter` stub registration happens via `import sgl_kernel` (see
   sgl_kernel._aiter_hcu_stub) for every AMD-only import on the HIP path.
"""

import torch

if getattr(torch.version, "hip", None) is not None:
    try:
        import vllm._custom_ops as _vco

        from sgl_kernel import rmsnorm as _sk_rmsnorm

        if not hasattr(torch.ops._C, "rms_norm"):

            def rms_norm(out, input, weight, epsilon):
                _sk_rmsnorm(input, weight, epsilon, out=out)

            def fused_add_rms_norm(
                out, input, residual_out, residual, weight, epsilon
            ):
                # sglang calls the 6-arg HIP form: out = rmsnorm(residual+input),
                # residual_out = residual+input. vllm's 4-arg in-place form is
                # bridged into it without extra copies.
                residual_in = residual if residual_out is residual else residual_out
                residual_in.copy_(residual)
                residual_in.add_(input)
                _sk_rmsnorm(residual_in, weight, epsilon, out=out)

            _vco.rms_norm = rms_norm
            _vco.fused_add_rms_norm = fused_add_rms_norm
    except Exception:  # pragma: no cover - vllm absent: sglang falls back natively
        pass

    try:
        import sgl_kernel as _sk

        if not hasattr(_sk, "apply_token_bitmask_inplace_cuda"):
            # sglang's xgrammar shim takes the HIP branch and imports this from
            # sgl_kernel; bridge it to the in-tree triton op (CUDA branch).
            from sglang.kernels.ops.grammar.bitmask_ops import (
                apply_token_bitmask_inplace_triton as _bitmask_triton,
            )

            _sk.apply_token_bitmask_inplace_cuda = _bitmask_triton
    except Exception:
        pass
