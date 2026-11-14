"""Scoped HCU implementations of the Core V4.1 mHC operations."""
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers import mhc as native_mhc

from hcu_adapter.kernels.mhc_post_triton import mhc_post
from .constructors import bind_constructor


class HcuDSV41PcpMHCPostOp(native_mhc.MHCPostOp):
    def forward_hip(self, x, residual, post_layer_mix, comb_res_mix):
        return mhc_post(x, residual, post_layer_mix, comb_res_mix)


class HcuDSV41PcpMHCPreDelayedOp(native_mhc.MHCPreDelayedOp):
    # Core's Torch gates avoid the unsupported HCU TileLang delayed seam.
    __init__ = bind_constructor(
        native_mhc.MHCPreDelayedOp.__init__,
        '82cd4838e10ee5ea00ed884145d38903350dbdff35129d745c0440a892ea3552',
        {'MHCPostOp': HcuDSV41PcpMHCPostOp})

    def forward_hip(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                    pre_mix=None, x=None, norm_weight=None, norm_eps=1e-6,
                    sublayer_out=None, post_layer_mix=None, comb_res_mix=None):
        residual = self._maybe_post(residual, sublayer_out, post_layer_mix, comb_res_mix)
        # Without pre_mix Core returns a free view; collapse with Core's Triton op.
        post_mix, comb_mix, layer_input, next_pre_mix = native_mhc.mhc_kernels.mhc_pre_delayed_torch(
            residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, pre_mix=None, x=x)
        if pre_mix is not None:
            layer_input = hc_collapse_triton(residual, pre_mix)
        return (residual, post_mix, comb_mix,
                native_mhc._apply_mhc_norm(layer_input, norm_weight, norm_eps), next_pre_mix)
