# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in DeepGEMM shifted mHC seam; no framework or vendor source changes.

Decoder RMSNorm is deliberately NOT taken from mega's y output.  The current
V4.1 delayed seam has norm_weight=None, so collapse must remain unnormalised.
Only the post + full-width projection case is admitted; model entry and all
other contracts are handled by the existing AITER runtime before dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
import inspect
import logging
from pathlib import Path
import weakref

import torch

LOG = logging.getLogger('hcu.adapter')

_PARAMETERS = (
    'x', 'residual', 'shifted_prev_mix', 'post_mix', 'comb_res_mix', 'fn',
    'mix_scales', 'mix_bases', 'hc_mult', 'hc_norm_eps', 'hc_pre_eps',
    'hc_post_scale', 'sinkhorn_eps', 'num_sinkhorn_iters', 'rmsnorm_weight',
    'rmsnorm_eps', 'rmsnorm_scale', 'new_residual', 'new_prev_mix',
    'new_post_mix', 'new_comb_res_mix', 'y_bf16', 'y_fp8', 'y_gemm_sf',
    'y_routed_sf', 'y_shared_sf', 'shared_sf_block_m', 'fn_packed',
)


@dataclass(frozen=True)
class MhcMegaProvider:
    bindings: tuple
    function: object
    prepare: object
    source_digest: str
    version: str

    def require_healthy(self):
        for owner, name, expected in self.bindings:
            if getattr(owner, name, None) is not expected:
                raise RuntimeError(f'DeepGEMM mHC provider changed after publication: {name}')

    def fingerprint(self):
        return dict(source=self.source_digest, version=self.version,
                    parameters=_PARAMETERS)


def load_provider():
    """Explicit selection requires both the documented Python ABI and real op."""
    module = import_module('deepgemm')
    api = import_module('deepgemm.mega_mhc')
    native = module.op
    function, prepare = module.mega_mhc, module.mega_mhc_prepare_fn
    bindings = tuple((owner, name, getattr(owner, name))
                     for owner in (module, api, native)
                     for name in ('mega_mhc', 'mega_mhc_prepare_fn'))
    if not all(callable(value) for _, _, value in bindings):
        raise RuntimeError('DeepGEMM mHC provider is not callable')
    if api.mega_mhc is not function or api.mega_mhc_prepare_fn is not prepare or api.op is not native:
        raise RuntimeError('DeepGEMM mHC Python/native bindings do not match')
    # Pybind functions do not expose inspect.signature. Check the public
    # wrapper's exact call form and the extension symbols it delegates to.
    signature = inspect.signature(function)
    prepare_signature = inspect.signature(prepare)
    if tuple(signature.parameters) != _PARAMETERS:
        raise RuntimeError('DeepGEMM mega_mhc ABI changed')
    if tuple(prepare_signature.parameters) != ('fn',):
        raise RuntimeError('DeepGEMM mega_mhc_prepare_fn ABI changed')
    try:
        signature.bind(*([None] * 21), y_bf16=None, fn_packed=None)
        prepare_signature.bind(None)
    except TypeError as error:
        raise RuntimeError('DeepGEMM mHC call ABI changed') from error
    bindings += ((module, 'op', native), (api, 'op', native))
    digest = hashlib.sha256()
    for owner in (module, api, native):
        digest.update(Path(owner.__file__).read_bytes())
    return MhcMegaProvider(bindings, function, prepare, digest.hexdigest(),
                           str(getattr(module, '__version__', 'unknown')))


class MhcMegaRuntime:
    """One mega call followed by the existing unnormalised BF16 collapse.

    Packed Fn is keyed by tensor identity AND its mutation version. Inference
    tensors without a version counter are packed on every call. Cached GPU
    buffers carry an event so first-use work remains safe across streams.
    """

    def __init__(self, provider, collapse):
        self.provider = provider
        self.collapse = collapse
        self._packed = {}
        self._ones = {}
        self._scratch = {}
        self.repack_count = 0
        self.reuse_count = 0
        self.versionless_repack_count = 0
        self._scratch_allocations = 0
        self._scratch_reuses = 0
        self._reported_versionless = False
        self._reported_cache = False

    def cache_stats(self):
        """Python dispatch counts; CUDA Graph replays do not execute Python."""
        return dict(repack=self.repack_count, reuse=self.reuse_count,
                    versionless=self.versionless_repack_count,
                    calls=self.repack_count + self.reuse_count,
                    scratch_allocations=self._scratch_allocations,
                    scratch_reuses=self._scratch_reuses)

    def _count_prepare(self, *, reused=False, versionless=False):
        if reused:
            self.reuse_count += 1
        else:
            self.repack_count += 1
        if versionless:
            self.versionless_repack_count += 1
            if not self._reported_versionless:
                self._reported_versionless = True
                LOG.warning(
                    'mHC mega Fn has no mutation version; repacking on every '
                    'such Python dispatch: repack=%d reuse=%d versionless=%d',
                    self.repack_count, self.reuse_count,
                    self.versionless_repack_count)
        calls = self.repack_count + self.reuse_count
        if calls >= 128 and not self._reported_cache:
            self._reported_cache = True
            LOG.info(
                'mHC mega Fn cache after warmup: repack=%d reuse=%d '
                'versionless=%d calls=%d; counts exclude CUDA Graph replays',
                self.repack_count, self.reuse_count,
                self.versionless_repack_count, calls)

    def _y_scratch(self, residual, stream):
        # Same-stream calls are ordered. Concurrent streams need distinct y
        # buffers even though y is discarded; do not introduce write races.
        tokens, _, hidden = residual.shape
        key = (residual.device, tokens, hidden)
        streams = self._scratch.setdefault(key, {})
        stream_id = stream.cuda_stream
        if stream_id not in streams:
            streams[stream_id] = residual.new_empty((tokens, hidden))
            self._scratch_allocations += 1
        else:
            self._scratch_reuses += 1
        # Retain these buffers until runtime teardown: a captured Graph may
        # still reference an eager-allocated scratch after Python returns.
        return streams[stream_id]

    @staticmethod
    def supported(residual, fn, scale, base, pre_mix, x, norm_weight,
                  sublayer_out, post_mix, comb_mix, sinkhorn_repeat):
        if (x is not None or norm_weight is not None or pre_mix is None
                or sublayer_out is None or post_mix is None or comb_mix is None
                or residual is None or residual.ndim != 3 or sinkhorn_repeat < 1):
            return False
        tokens, hc, hidden = residual.shape
        if (not tokens or hc != 4 or not 1024 <= hidden <= 32768 or hidden % 1024
                or fn.shape != (24, hc * hidden) or scale.shape != (3,)
                or base.shape != (24,) or pre_mix.shape != (tokens, hc)
                or sublayer_out.shape != (tokens, hidden)
                or post_mix.shape != (tokens, hc, 1)
                or comb_mix.shape != (tokens, hc, hc)):
            return False
        bf16 = (residual, sublayer_out)
        fp32 = (fn, scale, base, pre_mix, post_mix, comb_mix)
        return (all(t.dtype == torch.bfloat16 for t in bf16)
                and all(t.dtype == torch.float32 for t in fp32)
                and all(t.is_cuda and t.device == residual.device and t.is_contiguous()
                        for t in bf16 + fp32))

    @staticmethod
    def _ready(entry, stream):
        value, event, stream_id = entry
        if stream.cuda_stream != stream_id:
            stream.wait_event(event)
            # The weak Fn cache can be invalidated while another stream still
            # consumes this packed tensor. Track that use in the allocator too.
            value.record_stream(stream)
        return value

    @staticmethod
    def _record(value, stream):
        event = torch.cuda.Event()
        event.record(stream)
        return value, event, stream.cuda_stream

    def invalidate_packed_weights(self):
        """Discard derived weights before a supported host weight mutation."""
        self._packed.clear()

    def _prepare(self, fn, stream):
        try:
            version = fn._version
        except RuntimeError:
            # A versionless inference tensor can still be modified in-place.
            packed = self.provider.prepare(fn)
            self._count_prepare(versionless=True)
            return packed
        key = id(fn)
        found = self._packed.get(key)
        if found is not None and found[0]() is fn and found[1] == version:
            packed = self._ready(found[2], stream)
            self._count_prepare(reused=True)
            return packed
        packed = self.provider.prepare(fn)
        self._count_prepare()
        ref = weakref.ref(fn, lambda _ref: self._packed.pop(key, None))
        self._packed[key] = (ref, version, self._record(packed, stream))
        return packed

    def seam(self, residual: torch.Tensor, fn: torch.Tensor,
             hc_scale: torch.Tensor, hc_base: torch.Tensor,
             rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float,
             hc_post_mult_value: float, sinkhorn_repeat: int,
             pre_mix: torch.Tensor, sublayer_out: torch.Tensor,
             post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The caller checks the supported metadata before dispatch, and the
        # enclosing opaque op supplies the lifecycle/error latch.
        tokens, hc, hidden = residual.shape
        stream = torch.cuda.current_stream(residual.device)
        key = (residual.device, hidden)
        if key not in self._ones:
            weight = torch.ones(hidden, device=residual.device, dtype=torch.bfloat16)
            self._ones[key] = self._record(weight, stream)
        weight = self._ready(self._ones[key], stream)
        packed = self._prepare(fn, stream)
        updated = torch.empty_like(residual)
        next_pre = residual.new_empty((tokens, hc, 1), dtype=torch.float32)
        post = torch.empty_like(next_pre)
        comb = residual.new_empty((tokens, hc, hc), dtype=torch.float32)
        unused_y = self._y_scratch(residual, stream)
        self.provider.function(
            sublayer_out, residual, pre_mix.unsqueeze(-1), post_layer_mix,
            comb_res_mix, fn, hc_scale, hc_base, hc, rms_eps, hc_pre_eps,
            hc_post_mult_value, hc_sinkhorn_eps, sinkhorn_repeat, weight,
            1e-6, 1.0, updated, next_pre, post, comb,
            y_bf16=unused_y, fn_packed=packed)
        # Do not return mega's mandatory normalised y: weight=None in the
        # original delayed call means no decoder RMSNorm at this seam.
        layer = self.collapse(updated, pre_mix)
        return updated, post, comb, layer, next_pre.squeeze(-1)

    @staticmethod
    def seam_meta(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                  hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                  pre_mix, sublayer_out, post_layer_mix, comb_res_mix):
        tokens, hc, hidden = residual.shape
        return (torch.empty_like(residual),
                residual.new_empty((tokens, hc, 1), dtype=torch.float32),
                residual.new_empty((tokens, hc, hc), dtype=torch.float32),
                residual.new_empty((tokens, hidden)),
                residual.new_empty((tokens, hc), dtype=torch.float32))
