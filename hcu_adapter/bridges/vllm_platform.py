"""Minimal HCU platform identity for official vLLM v0.32."""

from vllm.platforms.rocm import RocmPlatform


class HcuPlatform(RocmPlatform):
    """Reuse official ROCm behavior under an HCU-specific platform identity."""

    device_name = "hcu"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
        num_heads=None,
    ) -> str:
        # Official v0.32 already ships TRITON_ATTN; HCU only supplies the
        # platform-specific default and delegates the complete selection.
        if selected_backend is None:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            selected_backend = AttentionBackendEnum.TRITON_ATTN
        return super().get_attn_backend_cls(
            selected_backend,
            attn_selector_config,
            num_heads,
        )
