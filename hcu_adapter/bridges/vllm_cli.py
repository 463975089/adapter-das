# SPDX-License-Identifier: Apache-2.0
"""Route out-of-tree vLLM CLI switches through the native additional config."""

import argparse
from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True)
class _Switch:
    flag: str
    key: str
    help: str
    validate: object


_lock = RLock()
_switches: dict[str, _Switch] = {}


def register_config_switch(flag, key, *, help, validate):
    if not flag.startswith("--") or not key or not callable(validate):
        raise ValueError("Invalid HCU CLI switch")
    switch = _Switch(flag, key, help, validate)
    with _lock:
        for registered in _switches.values():
            if registered.flag == flag or registered.key == key:
                if registered == switch:
                    return
                raise RuntimeError("Conflicting HCU CLI switch: " + flag)
        _switches[flag] = switch


def add_config_switches(parser):
    """Called on the engine parser by the upstream platform hook."""
    if parser is None or getattr(parser, "_hcu_cli_switches_added", False):
        return
    with _lock:
        switches = tuple(_switches.values())
    if not switches or parser.get_default("additional_config") is None:
        return
    bindings = {}
    for switch in switches:
        dest = "_hcu_cli_" + switch.key
        parser.add_argument(switch.flag, dest=dest, action="store_true",
                            default=argparse.SUPPRESS, help=switch.help)
        bindings[dest] = switch

    # Core's EngineArgs only accepts dataclass fields. Fold just this parser's
    # flags into a real field after parsing, so JSON merging is order independent.
    parse = parser.parse_known_args

    def parse_known_args(args=None, namespace=None):
        parsed, remaining = parse(args, namespace)
        for dest, switch in bindings.items():
            if not vars(parsed).pop(dest, False):
                continue
            extra = dict(parsed.additional_config or {})
            enabled = {"enabled": True}
            if switch.key in extra:
                current = extra[switch.key]
                if (not isinstance(current, dict) or current.keys() != enabled.keys()
                        or current["enabled"] is not True):
                    parser.error(f"{switch.flag} conflicts with --additional-config {switch.key}")
            extra[switch.key] = enabled
            parsed.additional_config = extra
        return parsed, remaining

    parser.parse_known_args = parse_known_args
    parser._hcu_cli_switches_added = True


def validate_config_switches(config):
    with _lock:
        switches = tuple(_switches.values())
    for switch in switches:
        switch.validate(config)
