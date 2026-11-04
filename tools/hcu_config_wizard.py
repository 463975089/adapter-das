# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Interactive config builder for the HCU adapter.

Prompts op by op, lists each selectable backend with its gate status,
and writes a minimal hcu_adapter.yaml containing ONLY the non-default
selections. This is the CLI stepping stone for the planned web UI --
both render from hcu_adapter.hcu_config.BACKEND_CATALOG, so a
new backend wired into the catalog shows up here (and in the UI)
automatically.

Usage:
    python tools/hcu_config_wizard.py            # interactive
    python tools/hcu_config_wizard.py --list     # catalog table, no writes
    python tools/hcu_config_wizard.py --show     # effective config now
    python tools/hcu_config_wizard.py --check F  # validate a config file
    python tools/hcu_config_wizard.py -o PATH    # non-default output path
    python tools/hcu_config_wizard.py --autotune # measure & pick tiers on
                                                 #   this machine (probe ->
                                                 #   judges -> A/B microbench
                                                 #   -> hcu_adapter.yaml +
                                                 #   autotune_report.json)
    python tools/hcu_config_wizard.py --autotune --dry
                                                 # ladder/catalog walkthrough,
                                                 # no subprocess judges
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from hcu_adapter import hcu_config  # noqa: E402

# production-status backends are what a bare Enter keeps
_ORDER = {"production": 0, "validated": 1, "candidate": 2, "broken": 3,
          "planned": 4}
_STATUS_TAG = {"production": "", "validated": " [validated]",
               "candidate": " [candidate]", "broken": " [BROKEN - see note]",
               "planned": " [planned - not wired]"}


def _defaults():
    out = {}
    for op, spec in hcu_config.BACKEND_CATALOG.items():
        prod = [b for b, s in spec["backends"].items()
                if s["status"] == "production"]
        if spec.get("var") or spec.get("vars"):
            # vars-compound: default is the first production backend
            out[op] = prod[0] if prod else None
        else:
            out[op] = prod[0] if prod else None
    return out


def list_catalog():
    print("HCU adapter backend catalog")
    print("=" * 72)
    for op, spec in sorted(hcu_config.BACKEND_CATALOG.items()):
        print(f"\n{op}  --  {spec['summary']}")
        items = sorted(spec["backends"].items(),
                       key=lambda kv: _ORDER.get(kv[1]["status"], 9))
        for name, meta in items:
            sel = " (selectable)" if meta["status"] != "planned" else ""
            print(f"   {name:<18}{_STATUS_TAG[meta['status']]}{sel}")
            if meta.get("note"):
                print(f"   {'':18}{meta['note']}")
    print("\nstatus legend: production=default gated | validated=gates "
          "passed, opt-in |")
    print("               candidate=wired, gate pending | planned=catalogued "
          "only (selecting raises)")


def show_effective():
    st = hcu_config.load()
    if st["path"] is None:
        print("no config file in effect (searched: "
              + ", ".join(hcu_config._candidate_paths()) + ")")
    else:
        print(f"config file: {st['path']}")
        for op, backend in sorted(st["ops"].items()):
            print(f"  {op}: {backend}")


def check_file(path):
    doc = hcu_config._parse_file(path)
    chosen, overrides = hcu_config._validate(doc, path)
    print(f"OK {path}: " + (" ".join(f"{k}={v}" for k, v in sorted(chosen.items()))
                            or "(all defaults, no ops set)"))
    for var, val in sorted(overrides.items()):
        print(f"  -> {var} = {val}")


def interactive(out_path):
    print("HCU adapter config wizard -- Enter keeps the default; only "
          "non-default choices are written.\n")
    defaults = _defaults()
    chosen = {}
    for op in sorted(hcu_config.BACKEND_CATALOG):
        spec = hcu_config.BACKEND_CATALOG[op]
        selectable = [b for b, m in spec["backends"].items()
                      if m["status"] != "planned"]
        if not selectable:
            continue
        default = defaults.get(op) or selectable[0]
        print(f"{op}: {spec['summary']}")
        for i, name in enumerate(selectable, 1):
            meta = spec["backends"][name]
            mark = " <default>" if name == default else ""
            print(f"  {i}) {name}{_STATUS_TAG[meta['status']]}{mark}")
            if meta.get("note"):
                print(f"     {meta['note']}")
        while True:
            raw = input(f"select [1-{len(selectable)}, default={default}]: ").strip()
            if raw == "":
                pick = default
                break
            if raw.isdigit() and 1 <= int(raw) <= len(selectable):
                pick = selectable[int(raw) - 1]
                break
            print("  invalid choice")
        if pick != default:
            chosen[op] = pick
        print()
    lines = ["version: 1", "ops:"]
    if chosen:
        for op in sorted(chosen):
            lines.append(f"  {op}: {chosen[op]}")
    else:
        lines[0] += "  # every op at production default; nothing to override"
    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {out_path}:")
    print(text)
    print("place it on the search path ($HCU_CONFIG, ./, or adapter root) "
          "and check the startup line 'HCU adapter config: ...'")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="print the catalog")
    ap.add_argument("--show", action="store_true",
                    help="print the config file currently in effect")
    ap.add_argument("--check", metavar="FILE",
                    help="validate a config file without loading it")
    ap.add_argument("-o", "--out", default="hcu_adapter.yaml",
                    help="wizard output path (default ./hcu_adapter.yaml)")
    ap.add_argument("--autotune", action="store_true",
                    help="measure tiers on this machine and write the yaml "
                         "+ autotune_report.json (probe -> judges -> A/B)")
    ap.add_argument("--dry", action="store_true",
                    help="with --autotune: no judge subprocesses; walk the "
                         "ladder/catalog only (config generation included)")
    ap.add_argument("--autotune-out", default="autotune_report.json",
                    help="report path for --autotune (default ./autotune_report.json)")
    args = ap.parse_args()
    if args.autotune:
        from tools._autotune import runner

        deep_gpu = not args.dry
        runner.run_autotune(out_yaml=args.out, out_report=args.autotune_out,
                            deep_gpu_check=deep_gpu, dry=args.dry)
    elif args.list:
        list_catalog()
    elif args.show:
        show_effective()
    elif args.check:
        check_file(args.check)
    else:
        interactive(args.out)


if __name__ == "__main__":
    main()
