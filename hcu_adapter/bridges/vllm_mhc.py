"""Opt-in vLLM 0.33 mHC compatibility via public OOT and official Torch math.

This host bridge owns registration/ABI only. Official post/head stay unchanged;
no vendor implementation, framework source rewrite or private registry is used.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from importlib import import_module
import inspect
import json
import os
from pathlib import Path

from packaging.version import Version

from hcu_adapter import hcu_env, registry
from hcu_adapter.kill_switch import op_disabled
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle

MODE_VARIABLE = 'HCU_VLLM_MHC_IMPL'
COMPILE_FACTOR = 'VLLM_HCU_ADAPTER_MHC_IMPLEMENTATION'
VALIDATED_ARCHES = frozenset({'gfx936'})


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


def selected_mode() -> str:
    if op_disabled('vllm_mhc'):
        return 'stock'
    value, _ = hcu_env.resolve(MODE_VARIABLE)
    if value not in ('stock', 'torch'):
        raise ValueError(f'{MODE_VARIABLE} must be stock or torch')
    return value


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
    if selected_mode() != 'torch':
        _refuse('mHC startup policy changed after registration; start a fresh process')
    for original, replacement in zip(p.originals, p.replacements):
        # Execute the public allocation/selection step without __init__, which
        # needs a current VllmConfig. No private OOT registry access is needed.
        if type(original.__new__(original)) is not replacement:
            _refuse('official mHC constructor selection changed after registration')
    if p.envs.environment_variables.get(COMPILE_FACTOR) is not p.factor_getter:
        _refuse('official mHC compilation factor ownership changed')


def require_healthy() -> None:
    """Read/check only; never publish or repair from a selector/load boundary."""
    _lifecycle.require_runtime_healthy()
    if _failure is not None:
        raise MhcCompatibilityError(_failure)
    p = _publication
    if p is None:
        if selected_mode() == 'torch':
            raise MhcCompatibilityError('mHC compatibility requested before runtime activation')
        return
    if p.pid != os.getpid():
        raise MhcCompatibilityError('mHC registration inherited from another PID; activate in this child first')
    try:
        _check_publication(p)
    except BaseException as error:
        _refuse('mHC registration health check failed', error)


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
        require_healthy()
        post, comb, layer = kernels.mhc_pre_torch(residual, fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits)
        if norm_weight is not None:
            layer = ir.ops.rms_norm(layer, norm_weight, norm_eps)
        return post, comb, layer

    def fused(x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor,
              fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
              rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float,
              sinkhorn_repeat: int, n_splits: int = 1, tile_n: int = 1,
              norm_weight: torch.Tensor | None = None, norm_eps: float = 0.0
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        require_healthy()
        updated = kernels.mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
        return (updated, *pre(updated, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits, norm_weight, norm_eps))

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


def install() -> bool:
    """Called only after the host's canonical version/platform/native guards."""
    global _publication
    if _failure is not None:
        raise MhcCompatibilityError(_failure)
    try:
        mode = selected_mode()
    except BaseException as error:
        if _publication is not None:
            _refuse('mHC policy invalid after registration; start a fresh process', error)
        raise
    if mode == 'stock' and _publication is None:
        return False

    def initialize():
        global _publication
        from hcu_adapter.bridges.vllm import detect_hcu, _installed_vllm_version
        detected = detect_hcu()
        if not detected.detected or detected.architecture not in VALIDATED_ARCHES:
            raise MhcCompatibilityError('official Torch mHC is only tested on real gfx936')
        parsed = Version(_installed_vllm_version())
        if parsed.epoch or parsed.release[:2] != (0,33):
            raise MhcCompatibilityError('official Torch mHC requires the vLLM 0.33 release line')
        if _publication is not None:
            _check_publication(_publication)
            _publication = replace(_publication, pid=os.getpid())
            return
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
        for original in (mhc.MHCPreOp, mhc.MHCFusedPostPreOp):
            if type(original.__new__(original)) is not original:
                raise MhcCompatibilityError('another owner already selects an mHC OOT implementation')
        if COMPILE_FACTOR in envs.environment_variables:
            raise MhcCompatibilityError('mHC compilation factor already has another owner')
        declarations = _declare(mhc, kernels, ir)
        files = (Path(__file__), Path(inspect.getfile(kernels)), Path(inspect.getfile(rms_ops)))
        content = dict(sources=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files],
                       torch=torch.__version__, hip=torch.version.hip)
        factor = 'official-torch:' + hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
        def factor_getter():
            return factor
        # Every operation below can publish permanent process state. An error,
        # including BaseException or partial registration, latches before text.
        try:
            for name, (function, fake, _) in zip(('hcu_adapter_mhc_pre','hcu_adapter_mhc_fused'), declarations):
                register(name, function, mutates_args=[], fake_impl=fake)
            mhc.MHCPreOp.register_oot(declarations[0][2])
            mhc.MHCFusedPostPreOp.register_oot(declarations[1][2])
            envs.environment_variables[COMPILE_FACTOR] = factor_getter
            _publication = _Publication(os.getpid(),(mhc.MHCPreOp,mhc.MHCFusedPostPreOp),
                tuple(item[2] for item in declarations),envs,factor_getter,factor)
            _check_publication(_publication)
            registry.register(op='vllm_mhc',impl='torch',target='vllm.model_executor.layers.mhc.MHCPreOp',
                mechanism=registry.MECH_SEAM,source=__name__,
                detail='candidate gfx936/v0.33 official Torch pre/post and public IR RMS via public OOT; post/head unchanged')
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
