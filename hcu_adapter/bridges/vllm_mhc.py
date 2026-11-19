# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in vLLM 0.33 mHC compatibility via public OOT and official Torch math.

This host bridge owns registration/ABI only. Official post/head stay unchanged;
no vendor implementation, framework source rewrite or private registry is used.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
from importlib import import_module, metadata
import inspect
import json
import logging
import os
from pathlib import Path

from packaging.version import Version

from hcu_adapter.infra import hcu_env
from hcu_adapter.infra import registry
from hcu_adapter.infra.kill_switch import op_disabled
from hcu_adapter.infra.runtime_lifecycle import ProcessLocalLifecycle

LOG = logging.getLogger("hcu.adapter")

MODE_VARIABLE = 'HCU_VLLM_MHC_IMPL'
COMPILE_FACTOR = 'VLLM_HCU_ADAPTER_MHC_IMPLEMENTATION'
SPLIT_VARIABLE = 'HCU_VLLM_MHC_PRE_SPLIT_K'
DELAYED_FUSION_VARIABLE = 'HCU_VLLM_MHC_DELAYED_FUSION'
MEGA_VARIABLE = 'HCU_VLLM_MHC_MEGA'
VALIDATED_ARCHES = frozenset({'gfx936', 'gfx938'})


class MhcCompatibilityError(RuntimeError):
    """The requested official mHC compatibility route is unavailable/unhealthy."""


_lifecycle = ProcessLocalLifecycle(runtime_action='vllm-mhc', error_type=MhcCompatibilityError)
_publication = None
_failure = None


@dataclass(frozen=True)
class _Publication:
    pid: int
    originals: tuple
    replacements: tuple
    envs: object
    factor_getter: object
    factor: str
    mode: str
    split_k: str
    delayed_fusion_requested: bool
    provider: object = None
    mega_provider: object = None
    mega_runtime: object = None


def selected_mode() -> str:
    if op_disabled('vllm_mhc'):
        return 'stock'
    value, source = hcu_env.resolve(MODE_VARIABLE, host='vllm')
    if value not in ('stock', 'torch', 'aiter'):
        raise ValueError(f'{MODE_VARIABLE} must be stock, torch or aiter')
    if source == 'default' and _publication is None:
        # General plugins run before model construction on every HCU host.
        # A feature default must not impose V4.1-only admission on that host.
        # Environment/config requests still go through the strict install checks.
        from hcu_adapter.bridges.vllm import detect_hcu, _installed_vllm_version
        detected = detect_hcu()
        if not detected.detected or detected.architecture not in VALIDATED_ARCHES:
            return 'stock'
        version = Version(_installed_vllm_version() or '0')
        if version.epoch or version.release[:2] != (0, 33):
            return 'stock'
    return value


def selected_split_k(mode):
    if mode != 'aiter':
        return '1'
    value, _ = hcu_env.resolve(SPLIT_VARIABLE)
    if value not in ('auto', '1', '2', '4', '8', '16', '32'):
        raise ValueError(f'{SPLIT_VARIABLE} must be auto or 1/2/4/8/16/32')
    return value


def selected_delayed_fusion(mode):
    if mode != 'aiter':
        return False
    value, _ = hcu_env.resolve(DELAYED_FUSION_VARIABLE, host='vllm')
    if value not in ('0', '1'):
        raise ValueError(f'{DELAYED_FUSION_VARIABLE} must be 0 or 1')
    return value == '1'


@lru_cache(maxsize=1)
def _supports_mega_default():
    from hcu_adapter.bridges.vllm import detect_hcu, _installed_vllm_version

    if _installed_vllm_version() != '0.33.0+dtk2610.torch2130.2611101131.g7364da':
        return False
    try:
        if metadata.version('deepgemm') != '2.1.0+dtk2610.torch2130.2611122142.g793351':
            return False
    except metadata.PackageNotFoundError:
        return False
    detected = detect_hcu()
    return detected.detected and detected.architecture == 'gfx938'


def selected_mega():
    if op_disabled('vllm_mhc'):
        return False
    value, source = hcu_env.resolve(MEGA_VARIABLE, host='vllm')
    if value not in ('0', '1'):
        raise ValueError(f'{MEGA_VARIABLE} must be 0 or 1')
    if value == '0':
        return False
    if source == 'default':
        return selected_mode() == 'aiter' and _supports_mega_default()
    return True


def _refuse(reason, error=None):
    global _failure
    # Publish before invoking exception formatting, including fork/reentry.
    if _failure is None:
        _failure = reason
        if error is not None:
            try:
                _failure = f'{reason}: {type(error).__name__}: {error}'
            except BaseException:
                pass
    raise MhcCompatibilityError(_failure) from error


def _check_publication(p):
    if (selected_mode() != p.mode or selected_split_k(p.mode) != p.split_k
            or selected_delayed_fusion(p.mode) != p.delayed_fusion_requested):
        _refuse('mHC startup policy changed after registration; start a fresh process')
    if selected_mega() != (p.mega_provider is not None):
        _refuse('mHC mega startup policy changed after registration; start a fresh process')
    _check_providers(p)
    for original, replacement in zip(p.originals, p.replacements):
        # Execute the public allocation/selection step without __init__, which
        # needs a current VllmConfig. No private OOT registry access is needed.
        if type(original.__new__(original)) is not replacement:
            _refuse('official mHC constructor selection changed after registration')
    if p.envs.environment_variables.get(COMPILE_FACTOR) is not p.factor_getter:
        _refuse('official mHC compilation factor ownership changed')


def _check_providers(p):
    if p.provider is not None:
        p.provider.require_healthy()
    if p.mega_provider is not None:
        p.mega_provider.require_healthy()


def require_healthy() -> None:
    """Validate startup policy and registration at lifecycle boundaries."""
    _lifecycle.require_runtime_healthy()
    if _failure is not None:
        raise MhcCompatibilityError(_failure)
    p = _publication
    if p is None:
        if selected_mode() in ('torch', 'aiter') or selected_mega():
            raise MhcCompatibilityError('mHC compatibility requested before runtime activation')
        return
    if p.pid != os.getpid():
        raise MhcCompatibilityError('mHC registration inherited from another PID; activate in this child first')
    try:
        _check_publication(p)
    except BaseException as error:
        _refuse('mHC registration health check failed', error)


def _check_dispatch_provider(provider) -> None:
    if _failure is not None:
        raise MhcCompatibilityError(_failure)
    if provider is not None:
        try:
            provider.require_healthy()
        except Exception as error:
            _refuse('mHC provider changed before dispatch; start a fresh process', error)


def check_graph_providers() -> None:
    """Check published providers before a Worker can replay captured work."""
    publication = _publication
    if publication is not None:
        _check_dispatch_provider(publication.provider)
        _check_dispatch_provider(publication.mega_provider)


def prepare_weight_mutation(vllm_config):
    """Protect cached Fn against writes that bypass PyTorch's version counter."""
    publication = _publication
    if publication is None or publication.mega_runtime is None:
        return
    runtime = publication.mega_runtime
    if runtime.repack_count and not vllm_config.model_config.enforce_eager:
        # The upstream update APIs do not invalidate captured graphs, which
        # may still hold the old packed Fn even if the Python cache is cleared.
        raise MhcCompatibilityError(
            'mHC Mega weight updates after Graph warmup require a fresh process')
    runtime.invalidate_packed_weights()


def _require_parameters(fn, names, defaults):
    parameters = inspect.signature(fn).parameters
    if tuple(parameters) != names or any(parameters[k].default != v for k, v in defaults.items()):
        raise MhcCompatibilityError(f'official mHC callable ABI changed: {fn.__name__}')


def _declare(mhc, kernels, ir):
    # Deferred so importing this bridge never imports Torch/vLLM/vendors.
    global torch
    torch = import_module('torch')

    def pre(residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
            rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float,
            sinkhorn_repeat: int, n_splits: int = 1, norm_weight: torch.Tensor | None = None,
            norm_eps: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _failure is not None:
            raise MhcCompatibilityError(_failure)
        post, comb, layer = kernels.mhc_pre_torch(residual, fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits)
        normed = layer
        if norm_weight is not None:
            normed = ir.ops.rms_norm(layer, norm_weight, norm_eps)
        return post, comb, normed

    def fused(x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor,
              fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
              rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float,
              sinkhorn_repeat: int, n_splits: int = 1, tile_n: int = 1,
              norm_weight: torch.Tensor | None = None, norm_eps: float = 0.0
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if _failure is not None:
            raise MhcCompatibilityError(_failure)
        updated = kernels.mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
        rest = pre(updated, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                   hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits,
                   norm_weight, norm_eps)
        return (updated, *rest)

    def pre_fake(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                 hc_post_mult_value, sinkhorn_repeat, n_splits=1, norm_weight=None, norm_eps=0.0):
        *outer, c, h = residual.shape
        return (residual.new_empty((*outer,c,1), dtype=torch.float32),
                residual.new_empty((*outer,c,c), dtype=torch.float32), residual.new_empty((*outer,h)))

    def fused_fake(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                   rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                   n_splits=1, tile_n=1, norm_weight=None, norm_eps=0.0):
        return (torch.empty_like(residual), *pre_fake(residual,fn,hc_scale,hc_base,rms_eps,
            hc_pre_eps,hc_sinkhorn_eps,hc_post_mult_value,sinkhorn_repeat,n_splits,norm_weight,norm_eps))

    class HcuMHCPreOp(mhc.MHCPreOp):
        def forward_hip(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                        hc_post_mult_value, sinkhorn_repeat, n_splits=1, norm_weight=None, norm_eps=0.0):
            return torch.ops.vllm.hcu_adapter_mhc_pre(residual,fn,hc_scale,hc_base,rms_eps,
                hc_pre_eps,hc_sinkhorn_eps,hc_post_mult_value,sinkhorn_repeat,n_splits,norm_weight,norm_eps)

    class HcuMHCFusedPostPreOp(mhc.MHCFusedPostPreOp):
        def forward_hip(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                        n_splits=1, tile_n=1, norm_weight=None, norm_eps=0.0):
            return torch.ops.vllm.hcu_adapter_mhc_fused(x,residual,post_layer_mix,comb_res_mix,
                fn,hc_scale,hc_base,rms_eps,hc_pre_eps,hc_sinkhorn_eps,hc_post_mult_value,
                sinkhorn_repeat,n_splits,tile_n,norm_weight,norm_eps)

    return (pre, pre_fake, HcuMHCPreOp), (fused, fused_fake, HcuMHCFusedPostPreOp)



def _declare_aiter(mhc, kernels, ir, backend, provider, split_k, mega=None, *, mega_provider=None):
    global torch
    torch = import_module('torch')
    runtime = backend.MhcAiterRuntime(provider, kernels, ir.ops.rms_norm,
        torch.ops.vllm.mhc_pre_mix_triton, torch.ops.vllm.hc_collapse_triton, split_k)
    from hcu_adapter.kernels.mhc_boundary_triton import supported as boundary_supported

    def execute(function, *args):
        _check_dispatch_provider(provider)
        try:
            return function(*args)
        except BaseException as error:
            _refuse('mHC AITER candidate execution failed; start a fresh process', error)

    def pre(residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
            rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float,
            sinkhorn_repeat: int, n_splits: int = 1, norm_weight: torch.Tensor | None = None,
            norm_eps: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return execute(runtime.pre, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits, norm_weight, norm_eps)

    def fused(x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor,
              fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
              rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float,
              sinkhorn_repeat: int, n_splits: int = 1, tile_n: int = 1,
              norm_weight: torch.Tensor | None = None, norm_eps: float = 0.0
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return execute(runtime.fused, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
            hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
            sinkhorn_repeat, n_splits, tile_n, norm_weight, norm_eps)

    def delayed(residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float,
                sinkhorn_repeat: int, pre_mix: torch.Tensor | None = None,
                x: torch.Tensor | None = None, norm_weight: torch.Tensor | None = None,
                norm_eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return execute(runtime.delayed, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix, x, norm_weight, norm_eps)

    def post(x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor,
             comb_res_mix: torch.Tensor) -> torch.Tensor:
        return execute(runtime.post, x, residual, post_layer_mix, comb_res_mix)

    def boundary(residual: torch.Tensor, sublayer_out: torch.Tensor,
                 post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor,
                 fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                 pre_mix: torch.Tensor | None, norm_weight: torch.Tensor | None,
                 norm_eps: float, rms_eps: float, hc_eps: float, post_mult: float, repeat: int
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        updated, post_mix, comb_mix, layer, next_pre = execute(
            runtime.boundary, residual, fn, hc_scale, hc_base, pre_mix,
            sublayer_out, post_layer_mix, comb_res_mix, rms_eps, hc_eps, hc_eps, post_mult, repeat)
        normed = execute(runtime._norm, layer, norm_weight, norm_eps)
        return updated, post_mix, comb_mix, normed, next_pre

    def pre_fake(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                 hc_post_mult_value, sinkhorn_repeat, n_splits=1, norm_weight=None, norm_eps=0.0):
        *outer, c, h = residual.shape
        return (residual.new_empty((*outer,c,1), dtype=torch.float32),
                residual.new_empty((*outer,c,c), dtype=torch.float32), residual.new_empty((*outer,h)))

    def fused_fake(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                   rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                   n_splits=1, tile_n=1, norm_weight=None, norm_eps=0.0):
        return torch.empty_like(residual), *pre_fake(residual,fn,hc_scale,hc_base,rms_eps,
            hc_pre_eps,hc_sinkhorn_eps,hc_post_mult_value,sinkhorn_repeat,n_splits,norm_weight,norm_eps)

    def delayed_fake(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                     hc_post_mult_value, sinkhorn_repeat, pre_mix=None, x=None, norm_weight=None, norm_eps=1e-6):
        *outer, c, _ = residual.shape
        return (*pre_fake(residual,fn,hc_scale,hc_base,rms_eps,hc_pre_eps,hc_sinkhorn_eps,
                          hc_post_mult_value,sinkhorn_repeat),
                residual.new_empty((*outer,c), dtype=torch.float32))

    def post_fake(x, residual, post_layer_mix, comb_res_mix):
        return torch.empty_like(residual)

    def boundary_fake(residual, sublayer_out, post_layer_mix, comb_res_mix,
                      fn, hc_scale, hc_base, pre_mix, norm_weight, norm_eps, rms_eps,
                      hc_eps, post_mult, repeat):
        m, c, h = residual.shape
        return (torch.empty_like(residual),
                residual.new_empty((m, c, 1), dtype=torch.float32),
                residual.new_empty((m, c, c), dtype=torch.float32),
                residual.new_empty((m, h)),
                residual.new_empty((m, c), dtype=torch.float32))

    class HcuMHCPreOp(mhc.MHCPreOp):
        def forward_hip(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                        hc_post_mult_value, sinkhorn_repeat, n_splits=1, norm_weight=None, norm_eps=0.0):
            return torch.ops.vllm.hcu_adapter_mhc_pre(residual,fn,hc_scale,hc_base,rms_eps,
                hc_pre_eps,hc_sinkhorn_eps,hc_post_mult_value,sinkhorn_repeat,n_splits,norm_weight,norm_eps)

    class HcuMHCFusedPostPreOp(mhc.MHCFusedPostPreOp):
        def forward_hip(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                        n_splits=1, tile_n=1, norm_weight=None, norm_eps=0.0):
            return torch.ops.vllm.hcu_adapter_mhc_fused(x,residual,post_layer_mix,comb_res_mix,
                fn,hc_scale,hc_base,rms_eps,hc_pre_eps,hc_sinkhorn_eps,hc_post_mult_value,
                sinkhorn_repeat,n_splits,tile_n,norm_weight,norm_eps)

    class HcuMHCPreDelayedOp(mhc.MHCPreDelayedOp):
        def forward_hip(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=None,
                        x=None, norm_weight=None, norm_eps=1e-6, sublayer_out=None,
                        post_layer_mix=None, comb_res_mix=None):
            if _failure is not None:
                raise MhcCompatibilityError(_failure)
            if mega is not None and mega.supported(
                    residual, fn, hc_scale, hc_base, pre_mix, x, norm_weight,
                    sublayer_out, post_layer_mix, comb_res_mix, sinkhorn_repeat):
                return torch.ops.vllm.hcu_adapter_mhc_mega_seam(
                    residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                    pre_mix, sublayer_out, post_layer_mix, comb_res_mix)
            if mega is not None:
                _check_dispatch_provider(mega_provider)
            if (x is None and split_k == 'auto' and provider.project is not None
                    and provider.tilelang_enabled and provider.tilelang_pre is not None
                    and boundary_supported(residual, fn, hc_scale, hc_base, pre_mix,
                        sublayer_out, post_layer_mix, comb_res_mix, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)):
                return torch.ops.vllm.hcu_adapter_mhc_boundary(
                    residual, sublayer_out, post_layer_mix, comb_res_mix,
                    fn, hc_scale, hc_base, pre_mix, norm_weight, norm_eps, rms_eps,
                    hc_pre_eps, hc_post_mult_value, sinkhorn_repeat)
            residual = self._maybe_post(residual,sublayer_out,post_layer_mix,comb_res_mix)
            return residual, *torch.ops.vllm.hcu_adapter_mhc_delayed(residual,fn,hc_scale,hc_base,
                rms_eps,hc_pre_eps,hc_sinkhorn_eps,hc_post_mult_value,sinkhorn_repeat,
                pre_mix,x,norm_weight,norm_eps)

    class HcuMHCPostOp(mhc.MHCPostOp):
        def forward_hip(self, x, residual, post_layer_mix, comb_res_mix):
            return torch.ops.vllm.hcu_adapter_mhc_post(x,residual,post_layer_mix,comb_res_mix)

    return (((pre, pre_fake, HcuMHCPreOp), (fused, fused_fake, HcuMHCFusedPostPreOp),
             (delayed, delayed_fake, HcuMHCPreDelayedOp), (post, post_fake, HcuMHCPostOp),
              (boundary, boundary_fake, None)),
            runtime)


def _declare_mega(runtime, mega_provider=None):
    def seam(residual: torch.Tensor, fn: torch.Tensor,
             hc_scale: torch.Tensor, hc_base: torch.Tensor,
             rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float,
             hc_post_mult_value: float, sinkhorn_repeat: int,
             pre_mix: torch.Tensor, sublayer_out: torch.Tensor,
             post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_dispatch_provider(mega_provider)
        if not runtime.supported(residual, fn, hc_scale, hc_base, pre_mix, None, None,
                                 sublayer_out, post_layer_mix, comb_res_mix, sinkhorn_repeat):
            raise MhcCompatibilityError('mHC mega seam called outside its supported contract')
        try:
            return runtime.seam(residual, fn, hc_scale, hc_base, rms_eps,
                hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                pre_mix, sublayer_out, post_layer_mix, comb_res_mix)
        except BaseException as error:
            _refuse('mHC mega seam execution failed; start a fresh process', error)
    return seam, runtime.seam_meta


def _load_aiter_provider():
    """Probe the real HIP provider; absent capability is documented fallback.

    A missing package selects the injected Torch reference before dispatch.
    A present but incompatible provider (broken import inside the module,
    non-callable symbol, drifted signature) is refused instead of downgraded.
    """
    backend = import_module('hcu_adapter.kernels.mhc_aiter')
    try:
        return backend.load_provider()
    except MhcCompatibilityError:
        raise
    except Exception as error:
        # Only an inspection failure is reclassified. KeyboardInterrupt and
        # SystemExit must propagate unchanged, or a real interruption reads as
        # "AITER mHC provider inspection failed" and the cause is hidden.
        raise MhcCompatibilityError(f'AITER mHC provider inspection failed: {error}') from error


def install() -> bool:
    """Called only after the host's canonical version/platform/native guards."""
    global _publication
    if _failure is not None:
        raise MhcCompatibilityError(_failure)
    try:
        mode = selected_mode()
        delayed_fusion_requested = selected_delayed_fusion(mode)
        mega_requested = selected_mega()
    except BaseException as error:
        if _publication is not None:
            _refuse('mHC policy invalid after registration; start a fresh process', error)
        raise
    # Stock stays untouched (pure upstream).
    if mode == 'stock' and _publication is None and not mega_requested:
        return False

    def initialize():
        global _publication
        from hcu_adapter.bridges.vllm import detect_hcu, _installed_vllm_version
        detected = detect_hcu()
        if not detected.detected or detected.architecture not in VALIDATED_ARCHES:
            raise MhcCompatibilityError(
                f'official Torch mHC requires real HCU arch in {sorted(VALIDATED_ARCHES)}; '
                f'got detected={detected.detected}, arch={detected.architecture!r}')
        parsed = Version(_installed_vllm_version())
        if parsed.epoch or parsed.release[:2] != (0,33):
            raise MhcCompatibilityError('official Torch mHC requires the vLLM 0.33 release line')
        if _publication is not None:
            _check_publication(_publication)
            _publication = replace(_publication, pid=os.getpid())
            return
        aiter_mode = mode == 'aiter'
        if mega_requested and (not aiter_mode or detected.architecture != 'gfx938'):
            raise MhcCompatibilityError('mHC mega seam requires aiter mode on gfx938')
        mhc = import_module('vllm.model_executor.layers.mhc')
        kernels = import_module('vllm.model_executor.kernels.mhc.torch')
        ir = import_module('vllm.ir')
        rms_ops = import_module('vllm.ir.ops.layernorm')
        envs = import_module('vllm.envs')
        register = import_module('vllm.utils.torch_utils').direct_register_custom_op
        pre_names = ('residual','fn','hc_scale','hc_base','rms_eps','hc_pre_eps','hc_sinkhorn_eps',
                     'hc_post_mult_value','sinkhorn_repeat','n_splits','norm_weight','norm_eps')
        defaults = dict(n_splits=1,norm_weight=None,norm_eps=0.0)
        _require_parameters(mhc.MHCPreOp.forward_hip, ('self',*pre_names), defaults)
        _require_parameters(mhc.MHCFusedPostPreOp.forward_hip,
            ('self','x','residual','post_layer_mix','comb_res_mix',*pre_names[1:10],'tile_n',*pre_names[10:]),
            dict(**defaults,tile_n=1))
        _require_parameters(kernels.mhc_pre_torch, pre_names[:10], dict(n_splits=1))
        _require_parameters(kernels.mhc_post_torch, ('x','residual','post_layer_mix','comb_res_mix'), {})
        # Both tiers wrap these two entries: deepseek_v41 calls MHCPreDelayedOp
        # and MHCPostOp, never MHCPreOp/MHCFusedPostPreOp.
        _require_parameters(mhc.MHCPreDelayedOp.forward_hip,
            ('self',*pre_names[:9],'pre_mix','x','norm_weight','norm_eps',
             'sublayer_out','post_layer_mix','comb_res_mix'),
            dict(pre_mix=None,x=None,norm_weight=None,norm_eps=1e-6,
                 sublayer_out=None,post_layer_mix=None,comb_res_mix=None))
        _require_parameters(mhc.MHCPostOp.forward_hip,
            ('self','x','residual','post_layer_mix','comb_res_mix'), {})
        for attribute in ('_maybe_post',):
            if not callable(getattr(mhc.MHCPreDelayedOp, attribute, None)):
                raise MhcCompatibilityError(f'official mHC delayed ABI changed: {attribute}')
        mega = None
        mega_provider = None
        if mega_requested:
            mega_backend = import_module('hcu_adapter.kernels.mhc_mega')
            mega_provider = mega_backend.load_provider()
            mega = mega_backend.MhcMegaRuntime(
                mega_provider, import_module('torch').ops.vllm.hc_collapse_triton)
        if aiter_mode:
            provider = _load_aiter_provider()
            split_k = selected_split_k(mode)
        else:
            provider = None
            split_k = '1'
        owners = ((mhc.MHCPreOp, mhc.MHCFusedPostPreOp, mhc.MHCPreDelayedOp, mhc.MHCPostOp)
                  if aiter_mode else (mhc.MHCPreOp, mhc.MHCFusedPostPreOp))
        for original in owners:
            if type(original.__new__(original)) is not original:
                raise MhcCompatibilityError('another owner already selects an mHC OOT implementation')
        if COMPILE_FACTOR in envs.environment_variables:
            raise MhcCompatibilityError('mHC compilation factor already has another owner')
        if aiter_mode:
            backend = import_module('hcu_adapter.kernels.mhc_aiter')
            declarations, runtime = _declare_aiter(
                mhc, kernels, ir, backend, provider, split_k, mega, mega_provider=mega_provider)
            names = ('hcu_adapter_mhc_pre','hcu_adapter_mhc_fused',
                     'hcu_adapter_mhc_delayed','hcu_adapter_mhc_post','hcu_adapter_mhc_boundary')
        else:
            declarations = _declare(mhc, kernels, ir)
            runtime = None
            names = ('hcu_adapter_mhc_pre','hcu_adapter_mhc_fused')
        files = (Path(__file__), Path(inspect.getfile(kernels)), Path(inspect.getfile(rms_ops)))
        if mega is not None:
            files += (Path(inspect.getfile(type(mega))),)
        if aiter_mode:
            source_dir = Path(__file__).parent.parent / 'kernels'
            files += (source_dir / 'mhc_aiter.py', source_dir / 'mhc_post_5120_triton.py',
                      source_dir / 'mhc_boundary_triton.py')
        content = dict(sources=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files],
                       torch=torch.__version__, hip=torch.version.hip,
                       mode=mode, split_k=split_k,
                       delayed_fusion_requested=delayed_fusion_requested,
                       delayed_fusion_sources=None,
                       provider=None if provider is None else provider.fingerprint(),
                       mega_requested=mega_requested,
                       mega=None if mega_provider is None else mega_provider.fingerprint())
        prefix = 'aiter' if aiter_mode else 'official-torch'
        impl = 'aiter' if aiter_mode and provider.available else 'torch'
        # Every operation below can publish permanent process state. An error,
        # including BaseException or partial registration, latches before text.
        try:
            for name, (function, fake, _) in zip(names, declarations):
                if function is not None:
                    register(name, function, mutates_args=[], fake_impl=fake)
            if mega is not None:
                mega_function, mega_fake = _declare_mega(mega, mega_provider)
                register('hcu_adapter_mhc_mega_seam', mega_function,
                         mutates_args=[], fake_impl=mega_fake)
            if (delayed_fusion_requested and provider.project is not None
                    and provider.stage2_ready):
                # Optional op: import/registration failures keep the two launches.
                try:
                    fused = import_module('hcu_adapter.kernels.mhc_fused_triton')
                    digests = [hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
                               for module in (backend, fused)]
                    register('hcu_adapter_mhc_pre_mix_collapse', fused.pre_mix_collapse,
                             mutates_args=[], fake_impl=fused.pre_mix_collapse_meta)
                    runtime.pre_mix_collapse = torch.ops.vllm.hcu_adapter_mhc_pre_mix_collapse
                except Exception as error:
                    LOG.warning(
                        'mHC fused gate/collapse unavailable; keeping two launches: %s', error)
                else:
                    content['delayed_fusion_sources'] = digests
            factor = prefix + ':' + hashlib.sha256(
                json.dumps(content, sort_keys=True).encode()).hexdigest()

            def factor_getter():
                return factor

            for owner, (_, _, replacement) in zip(owners, declarations):
                owner.register_oot(replacement)
            envs.environment_variables[COMPILE_FACTOR] = factor_getter
            _publication = _Publication(os.getpid(), owners,
                tuple(item[2] for item in declarations[:len(owners)]), envs, factor_getter, factor,
                mode, split_k, delayed_fusion_requested, provider, mega_provider, mega)
            _check_publication(_publication)
            detail = (f'candidate {detected.architecture}/v0.33 AITER HIP mHC '
                      f'pre/delayed/fused/post via public OOT; split-K={split_k}; '
                      f'TileLang-stage2={provider.tilelang_enabled}; provider={impl}; '
                      f'delayed-fusion={runtime.pre_mix_collapse is not None}; '
                      f'mega-shifted-seam={mega is not None}; '
                      f'gfx938 W4A8 eager validated; Graph and other models pending'
                      if aiter_mode else
                      f'candidate {detected.architecture}/v0.33 official Torch pre/post and '
                      f'public IR RMS via public OOT; post/head unchanged')
            registry.register(op='vllm_mhc', impl=impl,
                target='vllm.model_executor.layers.mhc.MHCPreOp',
                mechanism=registry.MECH_SEAM, source=__name__, detail=detail)
        except BaseException as error:
            _refuse('mHC publication failed; start a fresh process', error)

    def guarded_initialize():
        try:
            initialize()
        except BaseException as error:
            # Explicitly requested compatibility cannot proceed after partial
            # framework import or publication, even for BaseException.
            _refuse('mHC activation failed; start a fresh process', error)

    _lifecycle.run_once('vllm-mhc', guarded_initialize)
    require_healthy()
    return True
