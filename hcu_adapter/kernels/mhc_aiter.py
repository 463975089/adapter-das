# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

"""Framework-free AITER mHC provider and 029 staged delayed-Pre contract.

Adapted from vllm-plugin-das 01684cbb and 4b5d1e43 (Apache-2.0).
No host flags, Core methods, vendor functions or platform identities are changed.
Absent capabilities choose the injected Torch reference BEFORE execution. Import,
ABI and execution failures are never retried on a different implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
import inspect
import logging
from pathlib import Path

import torch

LOG = logging.getLogger("hcu.adapter")
_PRE = ('residual', 'fn', 'hc_scale', 'hc_base', 'rms_eps', 'hc_pre_eps',
        'hc_sinkhorn_eps', 'hc_post_mult_value', 'sinkhorn_repeat')
_SPECS = {
    'mhc_pre': _PRE,
    'mhc_post': ('out', 'x', 'residual', 'post_layer_mix', 'comb_res_mix'),
    'mhc_pre_gemm_sqrsum': ('out', 'sqrsum', 'x', 'fn', 'tile_k', 'use_tf32'),
    'mhc_pre_big_fuse': ('post_mix', 'comb_mix', 'layer_input', 'gemm_out_mul',
                         'gemm_out_sqrsum', 'hc_scale', 'hc_base', 'residual',
                         'rms_eps', 'hc_pre_eps', 'hc_sinkhorn_eps',
                         'hc_post_mult_value', 'sinkhorn_repeat'),
}


@dataclass(frozen=True)
class MhcAiterProvider:
    pre: object = None
    post: object = None
    project: object = None
    big_fuse: object = None
    module: object = None
    signatures: tuple = ()
    source_digest: str = 'absent'
    parameters: tuple = ()
    schemas: tuple = ()
    tilelang_pre: object = None
    tilelang_module: object = None
    tilelang_source_digest: str = 'disabled'
    tilelang_enabled: bool = False

    def signature_of(self, name):
        return dict(self.signatures).get(name, '')

    def parameters_of(self, name):
        return tuple(dict(self.parameters).get(name, ()))

    def require_healthy(self):
        """Read-only drift check; deliberately does no reflection here.

        Publication already recorded the authoritative signature and the raw
        registered schema. Rebuilding ``inspect.Signature``/``Parameter`` on
        every call both costs on a hot path and can be interrupted midway (a
        SIGTERM during shutdown landed inside that construction and surfaced as
        a false "provider changed" error). The hot path therefore only compares
        object identity plus the cheap schema string.
        """
        if not self.signatures:
            return
        for name, signature in self.signatures:
            function = getattr(self.module, name, None)
            expected = dict(zip(_SPECS, (self.pre, self.post, self.project, self.big_fuse))).get(name)
            if name == 'pre_big_fuse_tilelang':
                if not self.tilelang_enabled:
                    continue
                expected = self.tilelang_pre
            if function is not expected:
                raise RuntimeError(f'AITER mHC provider changed after publication: {name}')
            recorded = dict(self.schemas).get(name)
            if recorded is None:
                continue
            current = _schema_string(name)
            # An unreadable schema is not evidence of drift: the identity check
            # above already covers rebinding of the symbol itself.
            if current is not None and current != recorded:
                raise RuntimeError(f'AITER mHC provider changed after publication: {name}')

    def fingerprint(self):
        return dict(source=self.source_digest, signatures=self.signatures,
                    capabilities=[fn is not None for fn in
                                  (self.pre, self.post, self.project, self.big_fuse)],
                    tilelang_pre=self.tilelang_pre is not None,
                    tilelang_source=self.tilelang_source_digest,
                    tilelang_enabled=self.tilelang_enabled)

    @property
    def stage2_ready(self):
        """The selected stage-2 implementation is available.

        Enabling TileLang is an explicit rejection of the defective AITER
        big_fuse route, not merely an extra probe. If TileLang is unavailable,
        callers must choose the injected official Torch reference instead.
        """
        return (self.tilelang_pre is not None if self.tilelang_enabled
                else self.big_fuse is not None)

    @property
    def available(self):
        """True when at least one real HIP route can serve a call."""
        return self.pre is not None or self.post is not None or (
            self.project is not None and self.stage2_ready)


def _schema_string(name):
    """Registered torch schema text for an AITER op, or None when unreadable."""
    import torch
    try:
        return str(torch.ops.aiter.__getattr__(name).default._schema)
    except BaseException:
        return None


def _authoritative_signature(module, name, function):
    """Signature of a registered AITER op, taken from its torch schema.

    ``aiter.ops.mhc`` entries are wrapped by ``@compile_ops`` and
    ``@torch_compile_guard``, so ``inspect.signature`` only ever reports
    ``(*args, **kwargs)`` and cannot express the real ABI. The registered
    schema is authoritative; plain Python entry points that were never
    wrapped keep the reflective path.
    """
    import torch
    try:
        schema = torch.ops.aiter.__getattr__(name).default._schema
    except BaseException:
        return inspect.signature(function)
    parameters = []
    for argument in schema.arguments:
        # An absent schema default surfaces as None; none of the four mHC
        # entry points declares a genuine ``= None`` parameter.
        default = (inspect.Parameter.empty if argument.default_value is None
                   else argument.default_value)
        parameters.append(inspect.Parameter(
            argument.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default))
    return inspect.Signature(parameters)


def load_provider():
    """Inspect the real HIP provider once, before publishing any OOT class."""
    try:
        module = import_module('aiter.ops.mhc')
    except ModuleNotFoundError as error:
        if error.name not in ('aiter', 'aiter.ops', 'aiter.ops.mhc'):
            raise
        LOG.warning('AITER mHC package absent; selected aiter candidate uses official Torch')
        return MhcAiterProvider()
    functions = []
    signatures = []
    recorded = []
    schemas = []
    for name, prefix in _SPECS.items():
        function = getattr(module, name, None)
        if function is None:
            functions.append(None)
            LOG.warning('AITER mHC capability %s absent; this route uses official Torch', name)
            continue
        if not callable(function):
            raise RuntimeError(f'AITER mHC provider is not callable: {name}')
        signature = _authoritative_signature(module, name, function)
        parameters = tuple(signature.parameters)
        if parameters[:len(prefix)] != prefix:
            raise RuntimeError(f'AITER mHC provider ABI changed: {name}: {signature}')
        # The 029 HIP Pre may append use_tf32 and, in newer wheels, norm args.
        # Decoder RMS is deliberately kept outside the provider in both cases.
        tail = parameters[len(prefix):]
        if name == 'mhc_pre':
            if tail not in ((), ('use_tf32',), ('norm_weight', 'norm_eps'),
                            ('use_tf32', 'norm_weight', 'norm_eps')):
                raise RuntimeError(f'AITER mHC provider ABI changed: {name}: {signature}')
            kwargs = {'use_tf32': False} if 'use_tf32' in tail else {}
        else:
            if tail:
                raise RuntimeError(f'AITER mHC provider ABI changed: {name}: {signature}')
            kwargs = {}
        try:
            signature.bind(*([object()] * len(prefix)), **kwargs)
        except TypeError as error:
            raise RuntimeError(f'AITER mHC provider ABI changed: {name}: {signature}') from error
        functions.append(function)
        signatures.append((name, str(signature)))
        recorded.append((name, parameters))
        # Cache the raw schema text now: require_healthy() must not rebuild it.
        schema_text = _schema_string(name)
        if schema_text is not None:
            schemas.append((name, schema_text))
    module_source = getattr(module, '__file__', None)
    digest = hashlib.sha256(Path(module_source).read_bytes()).hexdigest() if module_source else 'no-file'

    # TileLang is a separate opt-in capability. Import failures mean unavailable;
    # they never substitute another implementation under the TileLang label.
    from hcu_adapter import hcu_env
    tilelang_enabled = hcu_env.enabled('VLLM_HCU_USE_AITER_TILELANG_MHC')
    tilelang_module = None
    tilelang_pre = None
    tilelang_digest = 'disabled'
    if tilelang_enabled:
        try:
            tilelang_module = import_module('aiter.ops.tilelang')
            candidate = getattr(tilelang_module, 'pre_big_fuse_tilelang', None)
            if callable(candidate):
                tilelang_pre = candidate
                tile_source = getattr(tilelang_module, '__file__', None)
                tilelang_digest = hashlib.sha256(Path(tile_source).read_bytes()).hexdigest() if tile_source else 'no-file'
            else:
                LOG.warning('AITER TileLang mHC enabled but pre_big_fuse_tilelang is absent; using Torch reference')
        except Exception as error:
            LOG.warning('AITER TileLang mHC enabled but unavailable; using Torch reference: %s', error)
            tilelang_module = None
    return MhcAiterProvider(*functions, module, tuple(signatures), digest,
                             tuple(recorded), tuple(schemas), tilelang_pre,
                             tilelang_module, tilelang_digest, tilelang_enabled)


def _gpu_tensors(tensors):
    return all(t.is_cuda and t.device == tensors[0].device and t.is_contiguous()
               for t in tensors)


def select_splits(tokens, hc, hidden, mode):
    if mode not in ('auto', '1', '2', '4', '8', '16', '32'):
        raise ValueError('mHC split-K must be auto or 1/2/4/8/16/32')
    splits = (32 if hc == 4 and hidden == 5120 and 0 < tokens <= 64 else 1) if mode == 'auto' else int(mode)
    width = hc * hidden
    if width % (64 * splits) or width // splits < 128:
        if mode == 'auto':
            return 1
        raise ValueError('mHC split-K does not divide the projection width or is too large')
    return splits


class MhcAiterRuntime:
    """029 HIP pre / delayed / post / composed post+pre, with injected refs."""

    def __init__(self, provider, kernels, rms_norm, pre_mix, collapse, split_k='auto'):
        if split_k not in ('auto', '1', '2', '4', '8', '16', '32'):
            raise ValueError('mHC split-K must be auto or 1/2/4/8/16/32')
        self.provider = provider
        self.kernels = kernels
        self.rms_norm = rms_norm
        self.pre_mix = pre_mix
        self.collapse = collapse
        self.split_k = split_k
        # Read the recorded schema ABI, not the (*args, **kwargs) wrapper.
        self._pre_tf32 = (provider.pre is not None
                          and 'use_tf32' in provider.parameters_of('mhc_pre'))

    def _norm(self, layer, weight, eps):
        return self.rms_norm(layer, weight, eps) if weight is not None and layer.numel() else layer

    @staticmethod
    def _empty_pre(residual):
        *outer, hc, hidden = residual.shape
        return (residual.new_empty((*outer, hc, 1), dtype=torch.float32),
                residual.new_empty((*outer, hc, hc), dtype=torch.float32),
                residual.new_empty((*outer, hidden)),
                residual.new_empty((*outer, hc), dtype=torch.float32))

    @staticmethod
    def _pre_supported(residual, fn, scale, base, pre_mix=None, x=None):
        if residual.ndim != 3 or residual.shape[1] != 4:
            return False
        tokens, hc, hidden = residual.shape
        width = hc * hidden if x is None else hidden
        tensors = [residual, fn, scale, base]
        if pre_mix is not None:
            if pre_mix.shape != (tokens, hc) or pre_mix.dtype != torch.float32:
                return False
            tensors.append(pre_mix)
        if x is not None:
            if x.shape != (tokens, hidden) or x.dtype != torch.bfloat16:
                return False
            tensors.append(x)
        return (hidden >= 1024 and hidden % 1024 == 0
                and residual.dtype == torch.bfloat16
                and fn.shape == (hc * (2 + hc), width)
                and scale.shape == (3,) and base.shape == (hc * (2 + hc),)
                and fn.dtype == scale.dtype == base.dtype == torch.float32
                and _gpu_tensors(tensors))

    def _staged_pre(self, residual, fn, scale, base, rms_eps, pre_eps,
                    sinkhorn_eps, post_mult, repeat, splits, x=None):
        tokens, hc, hidden = residual.shape
        projected = residual
        if x is not None:
            projected = x.unsqueeze(1).expand(-1, hc, -1).contiguous()
            expanded = fn.new_zeros(fn.shape[0], hc * hidden)
            expanded[:, :hidden].copy_(fn)
            fn = expanded
        # Output pitch is 32 while the contract exposes only 24 projection cols.
        padded = residual.new_empty((splits, tokens, 32), dtype=torch.float32)
        gemm = padded[:, :, :hc * (2 + hc)]
        sqrsum = residual.new_empty((splits, tokens), dtype=torch.float32)
        post, comb, layer, _ = self._empty_pre(residual)
        self.provider.project(gemm, sqrsum, projected, fn, 64, False)
        if self.provider.tilelang_enabled:
            if self.provider.tilelang_pre is None:
                raise RuntimeError('TileLang mHC stage2 was selected but is unavailable')
            self.provider.tilelang_pre(
                gemm.contiguous(), sqrsum, scale, base, residual,
                post.squeeze(-1), comb.flatten(1), layer, hidden, rms_eps,
                pre_eps, sinkhorn_eps, post_mult, repeat,
                n_splits=splits, mhc_mult=hc)
        else:
            if self.provider.big_fuse is None:
                raise RuntimeError('AITER mHC big_fuse was selected but is unavailable')
            self.provider.big_fuse(post, comb, layer, gemm, sqrsum, scale, base, residual,
                                   rms_eps, pre_eps, sinkhorn_eps, post_mult, repeat)
        return post, comb, layer, gemm, sqrsum

    def pre(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1,
            norm_weight=None, norm_eps=0.0):
        args = (residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
        if residual.numel() == 0:
            return self._empty_pre(residual)[:3]
        supported = self._pre_supported(residual, fn, hc_scale, hc_base)
        tilelang_ready = self.provider.tilelang_enabled and self.provider.tilelang_pre is not None
        t1_fallback = residual.shape[0] == 1 and not tilelang_ready
        native = (supported and not t1_fallback
                  and self.provider.pre is not None and n_splits == 1)
        staged = (supported and not t1_fallback and not native
                  and self.provider.project is not None and self.provider.stage2_ready)
        if native:
            with torch.device(residual.device):
                kwargs = {'use_tf32': False} if self._pre_tf32 else {}
                post, comb, layer = self.provider.pre(*args, **kwargs)
        elif staged:
            splits = select_splits(residual.shape[0], 4, residual.shape[-1], str(n_splits))
            post, comb, layer, _, _ = self._staged_pre(*args, splits)
        else:
            post, comb, layer = self.kernels.mhc_pre_torch(*args, n_splits)
        normed = self._norm(layer, norm_weight, norm_eps)
        return post, comb, normed

    def post(self, x, residual, post_layer_mix, comb_res_mix):
        if residual.numel() == 0:
            return torch.empty_like(residual)
        *outer, hc, hidden = residual.shape
        tensors = (x, residual, post_layer_mix, comb_res_mix)
        width_matches_history = hidden != 5120
        supported = (hc == 4 and hidden >= 1024 and hidden % 1024 == 0
                     and width_matches_history  # No historical Post v3 acceptance at this width.
                     and x.shape == (*outer, hidden)
                     and post_layer_mix.shape in ((*outer, hc), (*outer, hc, 1))
                     and comb_res_mix.shape == (*outer, hc, hc)
                     and residual.dtype in (torch.bfloat16, torch.float16)
                     and x.dtype == residual.dtype
                     and post_layer_mix.dtype == comb_res_mix.dtype == torch.float32
                     and _gpu_tensors(tensors))
        post_layer_mix = post_layer_mix.reshape(*outer, hc, 1)
        if self.provider.post is not None and supported:
            out = torch.empty_like(residual)
            self.provider.post(out, x, residual, post_layer_mix, comb_res_mix)
        else:
            out = self.kernels.mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
        return out

    def delayed(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                pre_mix=None, x=None, norm_weight=None, norm_eps=1e-6):
        args = (residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
        if residual.numel() == 0:
            return self._empty_pre(residual)
        tilelang_ready = self.provider.tilelang_enabled and self.provider.tilelang_pre is not None
        t1_fallback = residual.shape[0] == 1 and not tilelang_ready
        staged = (not t1_fallback and self.provider.project is not None
                  and self.provider.stage2_ready
                  and self._pre_supported(residual, fn, hc_scale, hc_base, pre_mix, x))
        if staged:
            hc, hidden = residual.shape[1:]
            splits = select_splits(residual.shape[0], hc, hidden, self.split_k)
            post, comb, _, gemm, sqrsum = self._staged_pre(*args, splits, x=x)
            next_pre = self.pre_mix(gemm, sqrsum, hc_scale, hc_base,
                                    hc, hc * hidden, rms_eps, hc_pre_eps)
            layer = residual[:, 0].clone() if pre_mix is None else self.collapse(residual, pre_mix)
        else:
            post, comb, layer, next_pre = self.kernels.mhc_pre_delayed_torch(
                *args, pre_mix=pre_mix, x=x)
            # The official no-pre_mix reference is a view of residual[:,0].
            layer = layer.clone()
        normed = self._norm(layer, norm_weight, norm_eps)
        return post, comb, normed, next_pre

    def fused(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
              hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
              sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None, norm_eps=0.0):
        # Preserve the BF16 Post storage boundary and decoder RMS exactly once.
        updated = self.post(x, residual, post_layer_mix, comb_res_mix)
        rest = self.pre(updated, fn, hc_scale, hc_base, rms_eps,
                        hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                        sinkhorn_repeat, n_splits, norm_weight, norm_eps)
        return updated, *rest
