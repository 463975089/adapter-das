"""Shared machinery for external-library meta-path adapters.

Strategy: prefer the REAL library whenever it actually exists on disk (the
DTK/HCU vendor wheel ports many of the ROCm triton kernels to HCU), and fall
back to per-module stubs only for the modules that are missing. `allowlist`
entries are *fill-in* overlays: after a real module executes, any allowlist
attribute the real module does not define itself is set on it, so our
validated bridges remain active even when the real package loads.
"""

import importlib.abc
import importlib.machinery
import sys
import types

# allowlist sentinel: means "this attr must NOT exist" — module __getattr__
# raises AttributeError so upstream feature-detection (`hasattr`) reports
# absent and takes its own fallback path. Use when reporting-broken is worse
# than reporting-absent (e.g. flashinfer allreduce_fusion API probes).
ABSENT = object()


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
            "__signature__",
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
            f"adapter entry (triton port) in hcu_adapter.{self._lib}."
        )

    def __mro_entries__(self, bases):
        # class-inheritance protocol: upstream CUDA-branch modules sometimes
        # subclass a missing symbol (e.g. flashinfer.comm.mnnvl.CommBackend).
        # Let the class statement close over object; constructed classes with
        # no real base raise AttributeError at first use of an inherited
        # method -- loud enough, and import must not break.
        import logging

        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: '%s' missing from %s -- subclassed as plain object;"
            " inherited-method calls will fail loudly (gap list)",
            self._name, self._lib,
        )
        return (object,)

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

    def __init__(self, inner, extras, lib, fullname, guards=None, overrides=None,
                 post_hooks=None):
        self._inner = inner
        self._extras = extras
        self._guards = dict(guards or {})
        self._overrides = dict(overrides or {})
        self._post_hooks = list(post_hooks or [])
        self._lib = lib
        self._fullname = fullname

    def create_module(self, spec):
        if self._inner is None:  # namespace package: default module creation
            return None
        return self._inner.create_module(spec)

    def exec_module(self, module):
        if self._inner is not None:  # namespace package: no code to exec
            self._inner.exec_module(module)
        for attr, value in self._extras.items():
            if value is ABSENT:
                # deny an attr a real module actually defines (feature probes)
                if hasattr(module, attr):
                    try:
                        delattr(module, attr)
                    except (AttributeError, TypeError):
                        setattr(module, attr, None)  # last resort: make it falsy-none
                continue
            if not hasattr(module, attr):
                setattr(module, attr, value)
        # replacements win over attrs the real module defines (unlike
        # extras, which only fill gaps): used by bridges that must take
        # over an upstream function wholesale (e.g. prefill-CP gate)
        for attr, value in self._overrides.items():
            setattr(module, attr, value)
        # arbitrary post-load hooks (e.g. wrapping class methods)
        for fn in self._post_hooks:
            try:
                fn(module)
            except Exception:
                pass
        if self._guards:
            from hcu_adapter import kill_switch as _ks

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

    def __init__(self, lib_name, allowlist=None, overrides=None, post_hooks=None,
                 deny=None):
        self.lib_name = lib_name
        # fullname -> absolute module attr -> value
        self.allowlist = dict(allowlist or {})
        # fullname -> attr -> value, force-set after load (replacements)
        self.overrides = dict(overrides or {})
        # fullname -> fn(module) run after exec (class-method wrapping etc.)
        self.post_hooks = dict(post_hooks or {})
        # fullnames that must NOT resolve: find_spec returns None so the real
        # import machinery raises ModuleNotFoundError (upstream except ImportError
        # fallbacks). Prefix match covers submodules.
        self._deny = tuple(deny or ())

    def _extra_attrs(self, fullname):
        return self.allowlist.get(fullname, {})

    def _override_attrs(self, fullname):
        return self.overrides.get(fullname, {})

    def _post_hooks_for(self, fullname):
        # values may be a single fn or an iterable of fns -> always a list
        ph = self.post_hooks.get(fullname, ())
        return [ph] if callable(ph) else list(ph)

    def _guard_attrs(self, fullname):
        """fullname -> {module attr: kill-switch op name}; subclasses override
        to guard real-wheel attrs (only applied while the op is disabled)."""
        return {}

    def find_spec(self, fullname, path=None, target=None):
        if not (
            fullname == self.lib_name or fullname.startswith(self.lib_name + ".")
        ):
            return None
        if any(fullname == d or fullname.startswith(d + ".") for d in self._deny):
            return None  # module must not exist -> upstream takes ImportError path
        try:
            real = importlib.machinery.PathFinder.find_spec(fullname, path)
        except Exception:
            real = None
        if real is not None:
            extras = self._extra_attrs(fullname)
            guards = self._guard_attrs(fullname)
            ov = self._override_attrs(fullname)
            ph = self._post_hooks_for(fullname)
            if extras or guards or ov or ph:
                real.loader = _OverlayLoader(
                    real.loader, extras, self.lib_name, fullname, guards, ov, ph
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
                v = _ok[item]
                if v is ABSENT:
                    raise AttributeError(
                        f"{_name}.{item} is explicitly absent on HCU"
                        " (adapter deny list)"
                    )
                return v
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
