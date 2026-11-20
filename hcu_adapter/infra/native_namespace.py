# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Private imported-callable bindings, preserving native function bodies."""
from types import FunctionType


def bind_native(function, substitutions):
    if not substitutions or any(name not in function.__code__.co_names
                                or name not in function.__globals__
                                for name in substitutions):
        raise RuntimeError('native callable anchor changed')
    namespace = dict(function.__globals__)
    namespace.update(substitutions)
    bound = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__annotations__ = function.__annotations__
    bound.__qualname__ = function.__qualname__
    return bound
