# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Prepared bindings for the serial SGLang startup sequence.

Resolve providers and aliases before entering this module. Registration can
fail, so publish only after every registration succeeds. The final dictionary
writes invoke no host descriptors or setters; keep prior values alive until
the group is complete, and undo our writes if publication is interrupted.
This is not a transaction for imports or concurrent host initialization.
"""
from dataclasses import dataclass

_MISSING = object()


@dataclass(frozen=True)
class Binding:
    namespace: dict
    name: str
    previous: object
    implementation: object

    @classmethod
    def prepare(cls, owner, name, implementation):
        namespace = vars(owner)
        if type(namespace) is not dict:
            raise TypeError("SGLang startup binding requires a writable namespace")
        return cls(namespace, name, namespace.get(name, _MISSING), implementation)


def publish(bindings, register):
    from hcu_adapter.infra import registry

    bindings = tuple(bindings)
    keys = [(id(binding.namespace), binding.name) for binding in bindings]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate SGLang startup binding")

    def check_owners():
        for binding in bindings:
            if binding.namespace.get(binding.name, _MISSING) is not binding.previous:
                raise RuntimeError("SGLang startup binding changed before publication")

    with registry.registration_transaction():
        check_owners()
        register()
        check_owners()
        try:
            for binding in bindings:
                binding.namespace[binding.name] = binding.implementation
        except BaseException:
            for binding in reversed(bindings):
                if binding.namespace.get(binding.name, _MISSING) is binding.implementation:
                    if binding.previous is _MISSING:
                        binding.namespace.pop(binding.name, None)
                    else:
                        binding.namespace[binding.name] = binding.previous
            raise
