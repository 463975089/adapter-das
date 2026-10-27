"""Host selection for the automatic HCU adapter bootstrap."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from pathlib import PurePath

HOST_ENV = "HCU_ADAPTER_HOST"
ACTIVE_HOST_ENV = "_HCU_ADAPTER_ACTIVE_HOST"

_HOSTS = frozenset({"sglang", "vllm"})
_PUBLIC_CHOICES = _HOSTS | {"auto", "none"}
_PROPAGATED_CHOICES = _HOSTS | {"none"}


def _choice(raw: object, *, variable: str, allowed: frozenset[str]) -> str:
    value = str(raw or "").strip().lower()
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{variable} must be one of: {choices}; got {raw!r}")
    return value


def _host_from_argv(argv: Sequence[str]) -> str | None:
    for index, raw in enumerate(argv):
        token = str(raw).strip().lower()
        basename = PurePath(token).name
        if basename in _HOSTS:
            return basename
        if token == "-m" and index + 1 < len(argv):
            module_root = str(argv[index + 1]).split(".", 1)[0].lower()
            if module_root in _HOSTS:
                return module_root
        module_root = token.split(".", 1)[0]
        if module_root in _HOSTS:
            return module_root
    return None


def _host_from_loaded(loaded_modules: Iterable[str]) -> str | None:
    roots = {str(name).split(".", 1)[0] for name in loaded_modules}
    present = roots & _HOSTS
    if len(present) == 1:
        return next(iter(present))
    if present == _HOSTS:
        return "sglang"
    return None


def detect_host(
    argv: Sequence[str],
    loaded_modules: Iterable[str],
    available: Iterable[str],
    environment: Mapping[str, str],
) -> str | None:
    """Select exactly one framework without importing either framework."""
    public = _choice(
        environment.get(HOST_ENV, "auto"),
        variable=HOST_ENV,
        allowed=_PUBLIC_CHOICES,
    )
    if public != "auto":
        return None if public == "none" else public

    inherited_raw = environment.get(ACTIVE_HOST_ENV)
    if inherited_raw:
        inherited = _choice(
            inherited_raw,
            variable=ACTIVE_HOST_ENV,
            allowed=_PROPAGATED_CHOICES,
        )
        return None if inherited == "none" else inherited

    intended = _host_from_argv(argv)
    if intended is not None:
        return intended

    loaded = _host_from_loaded(loaded_modules)
    if loaded is not None:
        return loaded

    present = set(available) & _HOSTS
    if len(present) == 1:
        return next(iter(present))
    if present == _HOSTS:
        return "sglang"
    return None


def propagate_host(host: str, environment: MutableMapping[str, str]) -> None:
    """Persist an automatic selection for spawn children without changing config."""
    selected = _choice(
        host,
        variable=ACTIVE_HOST_ENV,
        allowed=_PROPAGATED_CHOICES,
    )
    environment[ACTIVE_HOST_ENV] = selected
