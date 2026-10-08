"""Stub `aiter` package for HCU.

The upstream sglang quark/rocm shims do `if _is_hip: from aiter...` at module
level; on HCU (HIP device, AMD-only kernel library absent) that import hard
fails and blocks server startup for models that never touch those quant
schemes. This blank namespace satisfies the imports; any *call* into the stub
raises so real aiter-only paths stay loud, not silent.

Imported for its side effect from sgl_kernel.__init__ — must be imported
before any sglang module. Also installs a meta-path finder so nested
`aiter.xxx.yyy` imports work even for submodule paths not pre-registered.
"""

import importlib.abc
import importlib.machinery
import sys
import types


class _Missing:
    #: dunder/infrastructure lookups must fall through to normal module
    #: behaviour (inspect.getsourcefile walks __file__, pickle checks
    #: __path__, etc.); only real API access should raise.
    _PASS_THROUGH = frozenset(
        (
            "__file__",
            "__path__",
            "__spec__",
            "__loader__",
            "__package__",
            "__name__",
            "__doc__",
            "__all__",
            "__dir__",
            "__getattr__",
            "__code__",
            "__wrapped__",
        )
    )

    def __init__(self, name):
        self._name = name

    def __getattr__(self, item):
        if item in self._PASS_THROUGH:
            raise AttributeError(item)
        raise NotImplementedError(
            f"aiter stub on HCU: '{self._name}.{item}' is not available "
            "(AMD aiter kernels are not installed); this path is unsupported "
            "on HCU. If your model truly needs it, install a HCU-port of aiter."
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"aiter stub on HCU: '{self._name}' called; AMD aiter kernels are "
            "not available on HCU."
        )

    def __dir__(self):
        return []


class _AiterStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Pretend any `aiter[.sub...]` module exists; every attribute is missing.

    Allowlisted entries are *real* HCU implementations, not shims: a few aiter
    entry points used by sglang have device-agnostic semantics and are served
    by our triton kernels so HIP-platform code paths run (and are exercised by
    upstream parity tests) unmodified on HCU. Everything else raises on access.
    """

    PREFIX = "aiter"

    @staticmethod
    def _hcu_silu_and_mul(out, input, limit=0.0):
        """aiter's ROCm silu_and_mul: (out, input, limit); limit=0 = no clamp."""
        from sgl_kernel.elementwise import silu_and_mul

        silu_and_mul(input, out)
        if limit:
            out.clamp_(max=limit)
        return out

    @classmethod
    def _extra_attrs(cls, fullname):
        if fullname == "aiter.ops.triton.utils.types":
            import torch

            return {
                # fp8e4m3 on HCU torch; falls back if the dtype is absent
                "e4m3_dtype": getattr(torch, "float8_e4m3fn", torch.float16),
            }
        if fullname == "aiter.ops.triton.utils._triton.kernel_repr":
            return {
                "make_kernel_repr": lambda *a, **k: (lambda *ca, **ck: ""),
            }
        if fullname == "aiter":
            return {"silu_and_mul": cls._hcu_silu_and_mul}
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.PREFIX or fullname.startswith(self.PREFIX + "."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []  # mark as package so `aiter.x.y` imports resolve
        mod.__file__ = f"<aiter-stub:{spec.name}>"
        attrs = self._extra_attrs(spec.name) or {}

        def _getattr(item, _name=spec.name, _ok=attrs):
            if item in _ok:
                return _ok[item]
            return _Missing(f"{_name}.{item}")

        mod.__getattr__ = _getattr
        return mod

    def exec_module(self, module):
        pass


if not any(isinstance(f, _AiterStubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AiterStubFinder())
