# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Immutable HIP compiler inputs, outside the community SGLang source tree.

Only csrc/include assets are staged, never SGLang Python modules. Portability
transforms run in a private directory and are published as one content-addressed
bundle. The original compiler entry objects retain their aliases and route
relative/absolute sources and include paths to this bundle.
"""

import hashlib
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import types

LOG = logging.getLogger("hcu.adapter")
_LOCK = threading.RLock()
# Quant introduces the packed shuffle branch; warp then fixes its generic T.
_TRANSFORMS = (
    "jit_rope", "jit_store_cache", "jit_ptgq", "jit_dpskmeta",
    "jit_quant_fp8", "jit_warp", "jit_concat_mla", "jit_dot_product",
)
_MARKER = "_hcu_jit_input_binding"
_DISPATCH = "_hcu_jit_input_dispatch"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _community_root(root):
    """The community checkout root (nearest ancestor containing .git)."""
    for candidate in (root, *root.parents):
        if (candidate / ".git").exists():
            return candidate
    return root


def _assets(root):
    result = {}
    community = _community_root(root)
    for name in ("csrc", "include"):
        base = root / name
        if not base.is_dir() or base.is_symlink():
            raise RuntimeError("jit_sources: expected real csrc/include directories")
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                # Upstream organizes shared headers as in-repo symlinks (e.g.
                # sglang main: include/sgl_kernel/quantization/gguf/* ->
                # ../../../../../aot/...). Materialize those into the bundle
                # verbatim; only symlinks escaping the community checkout are
                # refused. read_bytes() below follows the link either way.
                target = path.resolve()
                if not (target == community or community in target.parents):
                    raise RuntimeError(
                        f"jit_sources: symbolic source asset escapes community tree: {path} -> {target}"
                    )
            # Some csrc subprojects ship standalone Python generators. They
            # are not compiler inputs and must not become a shadow Python tree.
            if path.is_file() and path.suffix not in (".py", ".pyc", ".pyo"):
                result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


def _hashes(assets):
    return {name: _digest(data) for name, data in sorted(assets.items())}


def _verify(bundle, expected):
    try:
        manifest = json.loads((bundle / "manifest.json").read_text())
        if manifest["key"] != expected:
            raise RuntimeError("key mismatch")
        actual = _hashes(_assets(bundle))
        forwards = {
            p.relative_to(bundle).as_posix(): _digest(p.read_bytes())
            for p in (bundle / "cuda-headers").rglob("*") if p.is_file()
        }
        if actual != manifest["outputs"] or forwards != manifest["forwarders"]:
            raise RuntimeError("content mismatch")
        if any(p.is_symlink() for p in bundle.rglob("*")):
            raise RuntimeError("symbolic bundle asset")
        return manifest
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise RuntimeError(f"jit_sources: invalid compiler bundle {bundle}: {exc}") from exc


def prepare(root, cache_root=None):
    """Snapshot actual compiler assets and publish all transformations atomically.

    Cache identity includes every input byte, every transform and the adapter
    forwarders. Corrupt existing artifacts are refused rather than overwritten.
    Concurrent publishers may duplicate work but never expose partial output.
    """
    root = Path(root).resolve()
    cache = Path(cache_root or Path.home() / ".cache/hcu_adapter/jit-inputs").resolve()
    community = next((p for p in (root, *root.parents) if (p / ".git").exists()), root)
    if cache == community or community in cache.parents:
        raise RuntimeError("jit_sources: cache must be outside the community JIT tree")
    modules = [importlib.import_module("hcu_adapter." + n) for n in _TRANSFORMS]
    from . import jit_compat

    assets = _assets(root)
    recipe = {
        "format": 1, "inputs": _hashes(assets),
        "transforms": {m.__name__: _digest(Path(m.__file__).read_bytes()) for m in modules},
        "manager": _digest(Path(__file__).read_bytes()),
        "forwarder_recipe": _digest(Path(jit_compat.__file__).read_bytes()),
        "forwarders": jit_compat._FORWARDS,
    }
    key = _digest(json.dumps(recipe, sort_keys=True).encode())
    bundle = cache / key
    cache.mkdir(parents=True, exist_ok=True)
    if bundle.exists():
        return bundle, _verify(bundle, key)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=cache))
    records = []

    def record(*args, **kwargs):
        kwargs["source"] = inspect.currentframe().f_back.f_globals["__name__"]
        # This target describes generated compiler data, not a community edit.
        kwargs["target"] = kwargs["target"].replace("text:", "jit-input:", 1)
        records.append([list(args), kwargs])

    try:
        for name, data in assets.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for module in modules:
            module._apply(stage, record)
        # Check allowlists/ops before publishing or binding a compiler entry.
        from . import registry, hcu_config
        for args, kwargs in records:
            if (args[0] not in hcu_config.BACKEND_CATALOG
                    or (args[0], kwargs["detail"]) not in registry.TEXT_PATCH_ALLOWLIST):
                raise RuntimeError("jit_sources: unreviewed compiler transformation")
        forwarders = {}
        for name, lines in jit_compat._FORWARDS.items():
            target = stage / "cuda-headers" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (jit_compat._MARK + "\n" + "\n".join(lines) + "\n").encode()
            target.write_bytes(data)
            forwarders[target.relative_to(stage).as_posix()] = _digest(data)
        manifest = {
            "key": key, "recipe": recipe, "source_root": str(root),
            "outputs": _hashes(_assets(stage)), "forwarders": forwarders,
            "records": records,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
        _verify(stage, key)
        # A source change during snapshot/transform must not become an apparent
        # clean snapshot. No writes, reset or cleanup of the community root.
        if _hashes(_assets(root)) != recipe["inputs"]:
            raise RuntimeError("jit_sources: community assets changed during preparation")
        try:
            os.rename(stage, bundle)
        except OSError:
            if not bundle.is_dir():
                raise
        return bundle, _verify(bundle, key)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _dispatch(*args, **kwargs):
    return _hcu_jit_input_dispatch(*args, **kwargs)


def _original(function, namespace):
    clone = types.FunctionType(
        function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def _check_function(function, names, identity):
    if not isinstance(function, types.FunctionType) or function.__closure__:
        raise RuntimeError("jit_sources: expected closure-free compiler entry")
    marker = getattr(function, _MARKER, None)
    if marker is not None:
        if (marker[0] != identity or function.__code__ is not _dispatch.__code__
                or function.__globals__.get(_DISPATCH) is not marker[1]):
            raise RuntimeError("jit_sources: compiler binding changed after installation")
        return False
    def referenced(code):
        result = set(code.co_names)
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                result.update(referenced(value))
        return result

    if _DISPATCH in function.__globals__ or not set(names) <= referenced(function.__code__):
        raise RuntimeError("jit_sources: compiler entry contract changed")
    return True


def _bind(function, impl, identity):
    function.__signature__ = inspect.signature(function)
    function.__globals__[_DISPATCH] = impl
    setattr(function, _MARKER, (identity, impl))
    function.__code__ = _dispatch.__code__


def bind(resolver, loader, source_root, bundle):
    """Retain original resolver and unwrapped load_jit function identities."""
    with _LOCK:
        source_root, bundle = Path(source_root).resolve(), Path(bundle).resolve()
        identity = (str(source_root), str(bundle))
        resolver_new = _check_function(resolver, ("KERNEL_PATH", "pathlib"), identity)
        loader_new = _check_function(loader, ("BuildSpec", "resolve_sources", "DEFAULT_INCLUDE"), identity)
        if resolver_new != loader_new:
            raise RuntimeError("jit_sources: partially installed compiler binding")
        if not resolver_new:
            return
        if list(inspect.signature(resolver).parameters) != ["files"]:
            raise RuntimeError("jit_sources: resolver signature changed")
        if "extra_include_paths" not in inspect.signature(loader).parameters:
            raise RuntimeError("jit_sources: load_jit include API changed")
        if loader.__globals__["resolve_sources"] is not resolver:
            raise RuntimeError("jit_sources: loader no longer uses the original resolver")
        if Path(resolver.__globals__["KERNEL_PATH"]).resolve() != source_root:
            raise RuntimeError("jit_sources: compiler source root changed")

        def mapped(value):
            path = Path(value)
            if path.is_absolute():
                resolved = path.resolve()
                for name in ("csrc", "include"):
                    base = source_root / name
                    if resolved == base or base in resolved.parents:
                        return str(bundle / resolved.relative_to(source_root))
            return str(value)

        resolve = _original(resolver, dict(resolver.__globals__, KERNEL_PATH=bundle))

        def resolve_inputs(files):
            if files is None:
                return resolve(None)
            # Retain upstream behavior for ../ paths and third-party sources
            # outside csrc/include instead of accidentally relocating them.
            paths = [
                mapped(Path(p) if Path(p).is_absolute()
                       else (source_root / "csrc" / p).resolve())
                for p in files
            ]
            return resolve(paths)

        includes = [mapped(p) for p in loader.__globals__["DEFAULT_INCLUDE"]]
        if str(bundle / "include") not in includes:
            raise RuntimeError("jit_sources: default compiler include root changed")
        includes.append(str(bundle / "cuda-headers"))
        build_spec = loader.__globals__["BuildSpec"]

        def compiler_spec(**fields):
            # Includes returned by registered dependency providers enter here
            # too, preserving the original live provider registry.
            fields["include_paths"] = tuple(mapped(p) for p in fields["include_paths"])
            return build_spec(**fields)

        load = _original(loader, dict(
            loader.__globals__, DEFAULT_INCLUDE=includes, resolve_sources=resolve_inputs,
            BuildSpec=compiler_spec,
        ))
        _bind(resolver, resolve_inputs, identity)
        _bind(loader, load, identity)


def install():
    """Install once per interpreter; spawn naturally builds/verifies its bundle."""
    with _LOCK:
        from sglang.kernels.jit.utils.compile import loader, spec
        from . import registry

        resolver = spec.resolve_sources
        raw_loader = inspect.unwrap(loader.load_jit)
        marker = getattr(resolver, _MARKER, None)
        root = Path(marker[0][0]) if marker else Path(resolver.__globals__["KERNEL_PATH"])
        bundle, manifest = prepare(root)
        bind(resolver, raw_loader, root, bundle)
        for args, kwargs in manifest["records"]:
            registry.register(*args, **kwargs)
        registry.register(
            "platform_detect", "immutable-jit-inputs",
            target="sglang.kernels.jit.utils.compile.loader.load_jit",
            mechanism=registry.MECH_REBIND,
            detail=f"compiler inputs {bundle}; community csrc/include untouched",
        )
        LOG.warning("HCU adapter: immutable JIT compiler inputs installed: %s", bundle)
        return bundle
