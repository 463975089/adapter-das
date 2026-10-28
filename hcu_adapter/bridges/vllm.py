"""vLLM-owned compatibility seams shared by framework host bridges."""


def install() -> None:
    """Keep host dispatch total until the vLLM integration MR lands."""


def install_rms_norm_bridge() -> bool:
    """Bind vLLM's HCU RMSNorm ABI to the adapter's equivalent kernels."""
    try:
        import vllm._custom_ops as custom_ops

        from sgl_kernel import rmsnorm
    except Exception:
        return False

    def rms_norm(out, input, weight, epsilon):
        rmsnorm(input, weight, epsilon, out=out)

    def fused_add_rms_norm(out, input, residual_out, residual, weight, epsilon):
        residual_destination = (
            residual if residual_out is residual else residual_out
        )
        residual_destination.copy_(residual)
        residual_destination.add_(input)
        rmsnorm(residual_destination, weight, epsilon, out=out)

    custom_ops.rms_norm = rms_norm
    custom_ops.fused_add_rms_norm = fused_add_rms_norm
    return True
