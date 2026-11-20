# SPDX-License-Identifier: Apache-2.0
"""Optional vendor KV placement callbacks at the official platform/worker seams.

Importing this registry is framework-free. Layout and ownership algorithms
belong to the registering extension. One owner supplies the matching engine
builder selector and worker pre-allocation transform atomically.
"""
from threading import RLock

_lock = RLock()
_provider = None


def register_kv_cache_layout(owner, select_builder, localize_config):
    global _provider
    if not isinstance(owner, str) or not owner or not callable(select_builder) or not callable(localize_config):
        raise TypeError('KV layout requires a named owner and two callbacks')
    entry = (owner, select_builder, localize_config)
    with _lock:
        if _provider is not None and _provider != entry:
            raise RuntimeError('KV cache layout already has a different owner')
        _provider = entry


def require_kv_cache_layout(owner, select_builder, localize_config):
    if _provider != (owner, select_builder, localize_config):
        raise RuntimeError('KV cache layout provider changed after activation')


def select_kv_cache_builder(config, default):
    if _provider is None:
        return default
    result = _provider[1](config)
    if result is None:
        return default
    if not isinstance(result, str) or not result:
        raise TypeError('KV cache builder selector must return a qualified class name or None')
    return result


def localize_kv_cache_config(config, cache_config):
    if _provider is None:
        return cache_config
    result = _provider[2](config, cache_config)
    if not isinstance(result, type(cache_config)):
        raise TypeError('KV cache localizer must preserve the config type')
    return result
