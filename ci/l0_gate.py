#!/usr/bin/env python3
"""L0 structural gate -- the no-GPU, no-deps, seconds-fast CI tier.

Runs on any machine (Gitee Go cloud runner, laptop, container). Verifies
the invariants that historically broke us (see docs/ci.md):

  HARD (exit 1 on violation)
    1. every .py parses (ast)                      -- catches half-edited MRs
    2. every .sh is LF-only                        -- CRLF kills bash silently
                                                     ( bitten 2026-10-11 )
    3. no secrets in tracked files                 -- gitee/TaskHub tokens
    4. every HCU_* var in hcu_env REGISTRY has a docs/env-vars.md row
    5. every var referenced by BACKEND_CATALOG exists in REGISTRY
    6. every sgl_kernel module sitecustomize imports resolves to a file
    7. hcu_adapter.example.yaml op keys are real catalog ops

  SOFT (warnings; --strict turns them into failures for local dev)
    - registered HCU_* vars read via bare os.environ outside hcu_env.py
      (env 三原则: behavioral reads must go through the registry)
    - .py files with CRLF endings

Usage:  python ci/l0_gate.py [--strict]
Exit:   0 green / 1 red; per-check PASS/FAIL lines go to stdout.
"""

import argparse
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", "build", "dist", ".zcode"}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def iter_files(*exts):
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(exts):
                yield os.path.join(dirpath, fn)


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


class Report:
    def __init__(self):
        self.fails = []
        self.warns = []

    def fail(self, check, msg):
        self.fails.append(f"[{check}] {msg}")

    def warn(self, check, msg):
        self.warns.append(f"[{check}] {msg}")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_py_syntax(r):
    n = 0
    for p in iter_files(".py"):
        n += 1
        try:
            src = open(p, "rb").read()
            ast.parse(src.decode("utf-8"))
            if b"\r\n" in src:
                r.warn("crlf", f"{rel(p)} has CRLF line endings")
        except SyntaxError as e:
            r.fail("syntax", f"{rel(p)}: {e}")
        except UnicodeDecodeError as e:
            r.fail("syntax", f"{rel(p)}: not utf-8 ({e})")
    return n


def check_shell_lf(r):
    n = 0
    for p in iter_files(".sh"):
        n += 1
        data = open(p, "rb").read()
        if b"\r\n" in data or b"\r" in data:
            r.fail("shell-lf", f"{rel(p)} has CR bytes (bash breaks silently)")
    return n


SECRET_PATTERNS = [
    (re.compile(rb"https://[^\s/@]+:[0-9a-f]{32}@"), "credential-in-url"),
    (re.compile(rb"[0-9a-f]{32}"), None),  # too broad alone; see below
]
# precise, cheap: the two real tokens ever in play + generic 32hex-with-context
SECRET_RULES = [
    (re.compile(rb"https://[^\s/@:\"]+:[0-9a-f]{32}@"), "gitee-token-in-url"),
    (re.compile(rb"(?i)(token|secret|password)\s*[=:]\s*['\"][0-9a-zA-Z]{20,}['\"]"), "literal-secret"),
]


def check_secrets(r):
    n = 0
    for p in iter_files(".py", ".sh", ".md", ".yml", ".yaml", ".json", ".cfg", ".toml", ".txt"):
        data = open(p, "rb").read()
        n += 1
        for pat, name in SECRET_RULES:
            m = pat.search(data)
            if m:
                line = data[: m.start()].count(b"\n") + 1
                r.fail("secrets", f"{rel(p)}:{line} {name}: {m.group(0)[:48].decode('utf-8','replace')}...")
    return n


def registry_names():
    src = open(os.path.join(ROOT, "sgl_kernel/_adapters/hcu_env.py"), encoding="utf-8").read()
    # registry keys look like    "HCU_XXX": {   at dict depth 1
    return set(re.findall(r'^\s{4}"(HCU_[A-Z0-9_]+)":\s*\{', src, re.M))


def catalog_var_refs():
    src = open(os.path.join(ROOT, "sgl_kernel/_adapters/hcu_config.py"), encoding="utf-8").read()
    refs = set(re.findall(r'"var":\s*"(HCU_[A-Z0-9_]+)"', src))
    for m in re.findall(r'"vars":\s*\{(.*?)\}', src, re.S):
        refs |= set(re.findall(r'"(HCU_[A-Z0-9_]+)"', m))
    return refs


def check_registry_docs(r, reg):
    doc = open(os.path.join(ROOT, "docs/env-vars.md"), encoding="utf-8").read()
    missing = [v for v in sorted(reg) if f"`{v}`" not in doc]
    for v in missing:
        r.fail("registry-docs", f"{v} not documented in docs/env-vars.md")


def check_catalog_refs(r, reg):
    refs = catalog_var_refs()
    missing = sorted(v for v in refs if v not in reg)
    for v in missing:
        r.fail("catalog-refs", f"{v} referenced by BACKEND_CATALOG but absent from REGISTRY")


def check_sitecustomize_imports(r):
    p = os.path.join(ROOT, "sitecustomize.py")
    src = open(p, encoding="utf-8").read()
    mods = set()
    for m in re.finditer(r"from (sgl_kernel(?:\._adapters)?(?:\.[\w.]+)?) import ([\w, ]+)", src):
        base = m.group(1)
        for name in m.group(2).split(","):
            name = name.strip().split(" as ")[0].strip()
            if not name or not name.isidentifier():
                continue
            mods.add((base, name))
    # symbols re-exported by the sgl_kernel package __init__: any word-level
    # mention means the import line lives there (multi-line paren imports too)
    init_src = open(os.path.join(ROOT, "sgl_kernel/__init__.py"), encoding="utf-8").read()
    init_words = set(re.findall(r"\b(\w+)\b", init_src))

    broken = []
    for base, name in sorted(mods):
        if base == "sgl_kernel":
            # package-level import: real submodule file OR __init__ re-export
            f = os.path.join(ROOT, "sgl_kernel", name + ".py")
            pkg = os.path.join(ROOT, "sgl_kernel", name, "__init__.py")
            if not (os.path.isfile(f) or os.path.isfile(pkg) or name in init_words):
                broken.append(f"{base}.{name}")
        else:
            path = os.path.join(ROOT, base.replace(".", "/") + ".py")
            pkgf = os.path.join(ROOT, base.replace(".", "/"), "__init__.py")
            if not (os.path.isfile(path) or os.path.isfile(pkgf)):
                broken.append(base)
    for mod in broken:
        r.fail("sitecustomize-imports", f"sitecustomize imports {mod} but no such module file")
    return len(mods)


def check_example_yaml(r):
    cat = open(os.path.join(ROOT, "sgl_kernel/_adapters/hcu_config.py"), encoding="utf-8").read()
    ops = set(re.findall(r'^\s{4}"([a-z0-9_]+)":\s*\{', cat, re.M))
    ex = open(os.path.join(ROOT, "hcu_adapter.example.yaml"), encoding="utf-8").read()
    in_ops = False
    for line in ex.splitlines():
        stripped = line.split("#")[0].rstrip()
        if re.match(r"^ops:\s*$", stripped):
            in_ops = True
            continue
        if in_ops:
            m = re.match(r"^\s{2}([a-z0-9_]+):\s*\S", stripped)
            if m and m.group(1) not in ops:
                r.fail("example-yaml", f"op '{m.group(1)}' not in BACKEND_CATALOG")
            if stripped and not line.startswith(" ") and stripped != "ops:":
                in_ops = False


def check_bare_environ(r, reg):
    """SOFT: registered vars read via os.environ outside the registry."""
    allowed_files = {
        "sgl_kernel/_adapters/hcu_env.py",  # the registry itself
        "ci/l0_gate.py",
    }
    pat = re.compile(r"os\.environ(?:\.get)?\(?[\"'](HCU_[A-Z0-9_]+)")
    for p in iter_files(".py"):
        rp = rel(p)
        if rp in allowed_files:
            continue
        try:
            src = open(p, encoding="utf-8").read()
        except Exception:
            continue
        for m in pat.finditer(src):
            var = m.group(1)
            if var in reg:
                line = src[: m.start()].count("\n") + 1
                r.warn(
                    "bare-environ",
                    f"{rp}:{line} reads {var} via os.environ; use hcu_env.resolve",
                )


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as failures")
    args = ap.parse_args()

    r = Report()
    n_py = check_py_syntax(r)
    n_sh = check_shell_lf(r)
    n_sec = check_secrets(r)
    reg = registry_names()
    if not reg:
        r.fail("registry", "could not parse any HCU_* names from hcu_env.py")
    check_registry_docs(r, reg)
    check_catalog_refs(r, reg)
    check_sitecustomize_imports(r)
    check_example_yaml(r)
    check_bare_environ(r, reg)

    print(f"l0 gate: scanned {n_py} py / {n_sh} sh / {n_sec} text files")
    for w in r.warns:
        print(f"WARN {w}")
    if r.fails:
        for f in r.fails:
            print(f"FAIL {f}")
        print(f"l0 gate: RED ({len(r.fails)} fail, {len(r.warns)} warn)")
        return 1
    if args.strict and r.warns:
        print(f"l0 gate: RED-strict ({len(r.warns)} warn)")
        return 1
    print(f"l0 gate: GREEN ({len(r.warns)} warn)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
