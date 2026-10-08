"""Shared machinery for external-library meta-path adapters."""

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


class LibStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Generic `<lib>[.sub...]` meta-path stub with per-module allowlists."""

    def __init__(self, lib_name, allowlist=None):
        self.lib_name = lib_name
        # fullname -> absolute module attr -> value (real implementations)
        self.allowlist = dict(allowlist or {})

    def _extra_attrs(self, fullname):
        return self.allowlist.get(fullname, {})

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.lib_name or fullname.startswith(self.lib_name + "."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

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
