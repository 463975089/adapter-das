"""Shared machinery for external-library meta-path adapters.

Strategy: prefer the REAL library whenever it actually exists on disk (the
DTK/DCU vendor wheel ports many of the ROCm triton kernels to HCU), and fall
back to per-module stubs only for the modules that are missing. `allowlist`
entries are *fill-in* overlays: after a real module executes, any allowlist
attribute the real module does not define itself is set on it, so our
validated bridges remain active even when the real package loads.
"""

import importlib.abc
import importlib.machinery
import sys
import types


class Missing:
    """Lazy failure handle: attribute access keeps returning handles (so
    `from lib import name` binds), any actual *call/number* raises loudly."""

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

    def __init__(self, name, lib):
        self._name = name
        self._lib = lib

    def __getattr__(self, item):
        if item in self._PASS_THROUGH:
            raise AttributeError(item)
        return Missing(f"{self._name}.{item}", self._lib)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"hcu-adapter '{self._lib}': '{self._name}' is not implemented on "
            f"HCU yet. The upstream code path that calls it needs a real "
            f"adapter entry (triton port) in sgl_kernel._adapters.{self._lib}."
        )

    def __dir__(self):
        return []

    def __repr__(self):
        return f"<hcu-adapter missing:{self._name}>"


class _OverlayLoader(importlib.abc.Loader):
    """Wraps a real module's loader; after exec:
    - fills allowlist attrs the real module does not define itself (our
      bridges win nothing else);
    - wraps attrs listed in `guards` (attr -> op name) with a kill-switch
      raiser when that op is disabled via HCU_DISABLE_OPS (see
      kill_switch.py). Guards only touch attrs the real module actually
      defines, so unknown wheel layouts are a no-op.
    """

    def __init__(self, inner, extras, lib, fullname, guards=None):
        self._inner = inner
        self._extras = extras
        self._guards = dict(guards or {})
        self._lib = lib
        self._fullname = fullname

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        for attr, value in self._extras.items():
            if not hasattr(module, attr):
                setattr(module, attr, value)
        if self._guards:
            from sgl_kernel._adapters import kill_switch as _ks

            for attr, op in self._guards.items():
                fn = getattr(module, attr, None)
                if callable(fn):
                    setattr(module, attr, _ks.guard(fn, op, attr))

    def __getattr__(self, item):
        return getattr(self._inner, item)


class LibStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Generic `<lib>[.sub...]` meta-path stub with per-module allowlists.

    Real modules on disk win (PathFinder only — no meta-path recursion);
    missing ones get a lazy stub package so `from lib import name` binds at
    import time, and calling an unimplemented entry raises a loud
    NotImplementedError naming the missing kernel.
    """

    def __init__(self, lib_name, allowlist=None):
        self.lib_name = lib_name
        # fullname -> absolute module attr -> value
        self.allowlist = dict(allowlist or {})

    def _extra_attrs(self, fullname):
        return self.allowlist.get(fullname, {})

    def _guard_attrs(self, fullname):
        """fullname -> {module attr: kill-switch op name}; subclasses override
        to guard real-wheel attrs (only applied while the op is disabled)."""
        return {}

    def find_spec(self, fullname, path=None, target=None):
        if not (
            fullname == self.lib_name or fullname.startswith(self.lib_name + ".")
        ):
            return None
        try:
            real = importlib.machinery.PathFinder.find_spec(fullname, path)
        except Exception:
            real = None
        if real is not None:
            extras = self._extra_attrs(fullname)
            guards = self._guard_attrs(fullname)
            if extras or guards:
                real.loader = _OverlayLoader(
                    real.loader, extras, self.lib_name, fullname, guards
                )
            return real
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__file__ = f"<{self.lib_name}-adapter:{spec.name}>"
        attrs = self._extra_attrs(spec.name)

        def _getattr(item, _name=spec.name, _ok=attrs, _lib=self.lib_name):
            if item in _ok:
                return _ok[item]
            return Missing(f"{_name}.{item}", _lib)

        mod.__getattr__ = _getattr
        return mod

    def exec_module(self, module):
        pass


def register_finder(finder) -> None:
    if not any(isinstance(f, type(finder)) and getattr(f, "lib_name", None) == finder.lib_name
               for f in sys.meta_path):
        sys.meta_path.insert(0, finder)


def is_registered(finder_cls, lib_name) -> bool:
    return any(
        isinstance(f, finder_cls) and getattr(f, "lib_name", None) == lib_name
        for f in sys.meta_path
    )
