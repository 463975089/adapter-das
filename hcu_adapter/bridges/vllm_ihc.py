# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""BoltOps iHC with Core's argument order and BF16 storage boundaries.

The Plugin owns model, device and size admission. Importing this module does
not import the vendor provider, Torch or vLLM.
"""


def validate_abi():
    import inspect
    from boltops.ihc import ihc_pre, ihc_post

    for function, expected in (
        (ihc_pre, ('residual', 'fn', 'scale', 'base', 'rms_eps', 'hc_eps', 'magnitude')),
        (ihc_post, ('x', 'residual', 'post')),
    ):
        if tuple(inspect.signature(function).parameters)[:len(expected)] != expected:
            raise RuntimeError('BoltOps iHC provider ABI changed')


def pre(x, weight, scale, base, magnitude, hc_eps, norm_eps):
    from boltops.ihc import ihc_pre

    return ihc_pre(x, weight, scale, base, norm_eps, hc_eps, magnitude)


def post_pre_rms_norm(x, residual, post, weight, scale, base, norm_weight,
                      magnitude, hc_eps, norm_eps):
    import torch
    from boltops.ihc import ihc_post
    import vllm._custom_ops as ops

    # Both stores round to BF16 before the following reduction, matching the
    # native chain. Do not fuse away these boundaries or modify input storage.
    updated = ihc_post(x, residual, post)
    reduced, gates = pre(updated, weight, scale, base, magnitude, hc_eps, norm_eps)
    output = torch.empty_like(reduced)
    ops.rms_norm(output, reduced, norm_weight, norm_eps)
    return output, gates, updated
