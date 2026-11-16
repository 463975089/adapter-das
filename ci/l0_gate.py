#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""L0 structural gate -- the no-GPU, no-deps, seconds-fast CI tier.

Runs on any machine (cloud runner, laptop, container). Verifies
the invariants that historically broke us (see docs/ci.md):

  HARD (exit 1 on violation)
    1. every .py parses (ast)                      -- catches half-edited MRs
    2. every .sh is LF-only                        -- CRLF kills bash silently
                                                     ( bitten 2026-10-11 )
    3. no secrets in tracked files                 -- remote tokens
    4. every HCU_* var in hcu_env REGISTRY has a docs/env-vars.md row
    5. every var referenced by BACKEND_CATALOG exists in REGISTRY
    6. every sgl_kernel module routing glue imports resolves to a file
    7. hcu_adapter.example.yaml op keys are real catalog ops
    8. no deep `sglang.kernels.ops.<domain>.<module>` imports in hcu_adapter
       without a bridge note (`JIT 唯一实现` / `替代档:`) -- package faces
       are the catalog's dispatch seam; module bodies pin one implementation.
       Both spellings count: `import sglang.kernels.ops.<d>.<m>` outright,
       and `from sglang.kernels.ops.<d> import <m>` when <m> is provably a
       module body (see check_ops_import_depth)

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
from pathlib import Path
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
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith("tmp")]
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
    (re.compile(rb"https://[^\s/@:\"]+:[0-9a-f]{32}@"), "remote-token-in-url"),
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
    src = open(os.path.join(ROOT, "hcu_adapter/hcu_env.py"), encoding="utf-8").read()
    # registry keys look like    "HCU_XXX": {   at dict depth 1
    return set(re.findall(r'^\s{4}"(HCU_[A-Z0-9_]+)":\s*\{', src, re.M))


def catalog_var_refs():
    src = open(os.path.join(ROOT, "hcu_adapter/hcu_config.py"), encoding="utf-8").read()
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


def routing_glue_files():
    """Return the thin loader, dispatcher, and framework bridge sources."""
    files = ["sitecustomize.py", "hcu_adapter/bootstrap.py"]
    bridge_dir = os.path.join(ROOT, "hcu_adapter", "bridges")
    for base, _, names in os.walk(bridge_dir):
        files.extend(rel(os.path.join(base, name)) for name in sorted(names) if name.endswith(".py"))
    return files


def check_routing_glue_imports(r):
    mods = set()
    for rp in routing_glue_files():
        src = open(os.path.join(ROOT, rp), encoding="utf-8").read()
        for m in re.finditer(
            r"from ((?:hcu_adapter|sgl_kernel)(?:\.[\w.]+)?) import ([\w, ]+)",
            src,
        ):
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
        r.fail("routing-imports", f"routing glue imports {mod} but no such module file")
    return len(mods)


def check_example_yaml(r):
    cat = open(os.path.join(ROOT, "hcu_adapter/hcu_config.py"), encoding="utf-8").read()
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


_OPS_DEEP_IMPORT = re.compile(r"^sglang\.kernels\.ops\.[^.]+\..+$")
_OPS_DOMAIN_IMPORT = re.compile(r"^sglang\.kernels\.ops\.[^.]+$")
_BRIDGE_NOTE_MARKERS = ("JIT 唯一实现", "替代档:")

# Same-named package-face re-exports: upstream ``quantization/__init__.py``
# re-exports the get_kernel-dispatching wrapper as ``per_token_group_quant``
# while a module of that name also exists. The 3-level
# ``from sglang.kernels.ops.quantization import per_token_group_quant`` binds
# the FACE (the sanctioned pattern); only the deep ``import`` spelling binds
# the module body. New dual-name discoveries get added here after a curator
# checks the host ``__init__`` actually re-exports the symbol.
_OPS_FACE_REEXPORTS = {("quantization", "per_token_group_quant")}


def _ops_body_module_proof():
    """(domain, module) pairs this repo itself binds as module bodies.

    Static evidence only: an actual import statement whose dotted path is
    ``sglang.kernels.ops.<domain>.<module>`` or deeper, anywhere in
    hcu_adapter (either spelling). Docstrings/comments are NOT evidence --
    raw-text harvesting would also pick up dotted symbol paths that name the
    sanctioned package-face pattern.
    """
    proof = set()
    for p in iter_files(".py"):
        if not rel(p).startswith("hcu_adapter/"):
            continue
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            else:
                continue
            for m in mods:
                parts = m.split(".")
                if (
                    len(parts) >= 5
                    and parts[:3] == ["sglang", "kernels", "ops"]
                    and parts[3] and parts[4]
                ):
                    proof.add((parts[3], parts[4]))
    return proof


def check_ops_import_depth(r):
    """HARD: hcu_adapter must not pin a host module-body implementation.

    ``sglang.kernels.ops.<domain>`` is the catalog / fused-op dispatch seam;
    importing ``…<domain>.<module>`` bypasses the priority + capability filter
    and pins that module's implementation (on HCU this forced a 33s ninja JIT
    build for ``silu_and_mul`` in the L1 kernels segment). Exceptions must say
    why inline: ``JIT 唯一实现`` (no other tier exists) or ``替代档: <location>``
    (point at the alternative tier this call shape cannot use / does not need).

    Equivalent spellings are checked too (2026-11-12 review): plain
    ``import sglang.kernels.ops.<domain>.<module>`` (ast.Import) always binds
    the module body. ``from sglang.kernels.ops.<domain> import <module>``
    binds a body when the name repeats the domain
    (``from …activation import activation``) or is proven a module by this
    repo's own deep imports -- minus package faces that re-export a
    same-named wrapper symbol (_OPS_FACE_REEXPORTS): those 3-level imports
    bind the sanctioned face, not the body.
    """
    proof = _ops_body_module_proof()
    n = 0
    for p in iter_files(".py"):
        rp = rel(p)
        if not rp.startswith("hcu_adapter/"):
            continue
        try:
            src = open(p, encoding="utf-8").read()
            tree = ast.parse(src)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue  # check_py_syntax owns that failure
        lines = src.splitlines()
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hits.extend(
                    (node, a.name)
                    for a in node.names
                    if _OPS_DEEP_IMPORT.match(a.name)
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if _OPS_DEEP_IMPORT.match(node.module):
                    hits.append((node, node.module))
                elif _OPS_DOMAIN_IMPORT.match(node.module):
                    domain = node.module.rsplit(".", 1)[-1]
                    for a in node.names:
                        if a.name == domain or (
                            (domain, a.name) in proof
                            and (domain, a.name) not in _OPS_FACE_REEXPORTS
                        ):
                            hits.append(
                                (node, f"{node.module}.{a.name} (from-import)")
                            )
        for node, what in hits:
            n += 1
            window = "\n".join(lines[max(0, node.lineno - 3): node.end_lineno])
            if not any(marker in window for marker in _BRIDGE_NOTE_MARKERS):
                r.fail(
                    "ops-import-depth",
                    f"{rp}:{node.lineno} imports host module body "
                    f"{what!r} without a bridge note (add "
                    f"'JIT 唯一实现' or '替代档: ...')",
                )
    return n


def check_bare_environ(r, reg):
    """SOFT: registered vars read via os.environ outside the registry."""
    allowed_files = {
        "hcu_adapter/hcu_env.py",  # the registry itself
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


# --- governance rules (docs/governance.md §3/§4) ----------------------------
# The glue layer routes, the kernel library computes. These three keep the
# boundary machine-checked so the team can grow without re-blurring it.

NUMERIC_CODE_PATTERNS = [
    (re.compile(r"^\s*@triton\.jit", re.M), "@triton.jit kernel"),
    (re.compile(r"^\s*import\s+triton\b", re.M), "import triton"),
    (re.compile(r"\btl\.\w+"), "triton language reference"),
]


def check_glue_has_no_numerics(r):
    """HARD: host routing glue has no embedded Triton operator math.

    Kernels belong in sgl_kernel/*.py. Diagnosing 'which implementation
    served this tensor' is impossible when the glue layer itself computes.
    """
    for rp in routing_glue_files():
        p = os.path.join(ROOT, rp)
        if not os.path.isfile(p):
            continue
        src = open(p, encoding="utf-8").read()
        # strip comments so root-cause notes can name the mechanism
        stripped = "\n".join(
            ln.split("#")[0] if not ln.lstrip().startswith("#") else ""
            for ln in src.split("\n")
        )
        for pat, what in NUMERIC_CODE_PATTERNS:
            m = pat.search(stripped)
            if m:
                line = stripped[: m.start()].count("\n") + 1
                r.fail("glue-numerics", f"{rp}:{line} has {what}; highway "
                      f"for kernels is sgl_kernel/ (governance §3)")


def _parse_allowlist():
    """Text-patch needles allowed, parsed from registry.py (no import)."""
    src = open(os.path.join(
        ROOT, "hcu_adapter/registry.py"), encoding="utf-8").read()
    body = re.search(
        r"TEXT_PATCH_ALLOWLIST\s*=\s*\{(.*?)\n\}", src, re.S)
    if not body:
        return set(), False
    allow = True
    needles = set()
    for op, needle in re.findall(r'\("([\w_]+)",\s*"((?:[^"\\]|\\.)*)"\)', body.group(1)):
        needles.add((op, needle))
    return needles, allow


def check_text_patch_allowlist(r):
    """HARD: source-replace recompiles are allowlisted in registry.py.

    Every `assert _is_cuda or _is_xpu`-style needle patched in routing
    glue must appear in the registry's TEXT_PATCH_ALLOWLIST
    (governance review each time this ladder rung is used). We count the
    literal `_is_cuda or _is_xpu` replace targets -- the only pattern in
    play; new needles must be added to the allowlist AND this check.
    """
    needles, ok = _parse_allowlist()
    if not ok:
        r.fail("text-patch", "cannot parse TEXT_PATCH_ALLOWLIST from "
               "hcu_adapter/registry.py")
        return
    allowed_needles = {n for _, n in needles}
    sources = []
    for rp in routing_glue_files():
        src = open(os.path.join(ROOT, rp), encoding="utf-8").read()
        sources.append((rp, src))
        # candidate needles: any quoted string starting with `assert _is_cuda`
        # that a .replace patch is feeding (first arg of .replace( — the
        # upstream text being replaced)
        for m in re.finditer(
                r"\.replace\(\s*[rbf]?['\"]([^'\"]*)['\"]", src):
            needle = m.group(1)
            if "_is_cuda" not in needle:
                continue
            # allowlist entries are the `_old` constant strings upstream uses
            if not any(needle in a or a in needle for a in allowed_needles):
                line = src[: m.start()].count("\n") + 1
                r.fail("text-patch", f"{rp}:{line} patches source text "
                       f"{needle!r} not in registry TEXT_PATCH_ALLOWLIST")
    # count distinct patch installs vs allowlist size: allowlist shrinks
    # only via governance review; hidden extra sites fail the needle scan
    n_installs = sum(
        len(re.findall(r"\.replace\(\s*\"", src)) for _, src in sources
    )
    if n_installs > len(needles):
        r.warn("text-patch", f"routing glue has {n_installs} .replace() sites "
               f"vs {len(needles)} allowlisted needles -- review")


def check_registry_ops_in_catalog(r):
    """HARD: every registry.register(op=...) names a BACKEND_CATALOG op."""
    from hcu_adapter.hcu_config import BACKEND_CATALOG
    ops = set(BACKEND_CATALOG)
    for p in iter_files(".py"):
        rp = rel(p)
        if (not rp.startswith("hcu_adapter/") and "/bridges/" not in rp
                and rp != "sitecustomize.py"):
            continue
        try:
            src = open(p, encoding="utf-8").read()
        except Exception:
            continue
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "registry" or node.func.attr != "register":
                continue
            value = node.args[0] if node.args else next((x.value for x in node.keywords if x.arg == "op"), None)
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value not in ops:
                r.fail("registry-ops", f"{rp}:{node.lineno} registers unknown op {value.value!r}")



# ---------------------------------------------------------------------------

# governance §3b: operator implementations, actual dispatch facades and mechanisms.
_KERNEL_TIERS = {"emu", "triton", "cpp", "fork", "ref", "aiter", "lightop", "dtk", "rccl"}
_KERNEL_FACADES = {"gemm", "gather", "norm", "index_ops", "fused_transform"}
_KERNEL_MECHANISMS = {"runtime_config", "lightop_usage"}


def check_impl_naming(r):
    """HARD: module-naming rules from governance §3b.

    a) `_ref` modules (diagnostic references, governance §2) must never be
       imported from production code (hcu_adapter / sgl_kernel / bridges /
       sitecustomize);
    b) platform words (hip/cuda/rocm) are not tiers -- banned as filename
       tier suffixes for implementation modules;
    c) docs/tier-matrix.md must be freshly rendered from BACKEND_CATALOG.
    """
    import subprocess

    for p in iter_files(".py"):
        rp = rel(p)
        in_prod = (rp.startswith(("hcu_adapter/", "sgl_kernel/"))
                   or rp == "sitecustomize.py")
        try:
            src = open(p, encoding="utf-8").read()
        except Exception:
            continue
        for m in re.finditer(
                r"^\s*(?:from|import)\s+[\w.]*?(\w*_ref)\b", src, re.M):
            if m.group(1) == "weak_ref":
                continue  # torch weak-tensor helper, not a diagnostic reference
            if in_prod:
                line = src[: m.start()].count("\n") + 1
                r.fail("impl-naming", f"{rp}:{line} production code imports "
                       f"diagnostic reference module '{m.group(1)}' "
                       f"(governance §2/§3b: _ref stays under tests/)")
            elif not rp.startswith("tests/"):
                r.warn("impl-naming", f"{rp} imports a _ref module outside "
                       f"tests/ ({m.group(1)}) -- verify intent")
        if rp.startswith("hcu_adapter/kernels/"):
            stem = os.path.splitext(os.path.basename(rp))[0]
            if (stem != "__init__" and stem not in _KERNEL_FACADES
                    and stem not in _KERNEL_MECHANISMS
                    and stem.rsplit("_", 1)[-1] not in _KERNEL_TIERS):
                r.fail("impl-naming", f"{rp}: expected an operator tier suffix or "
                       "a documented facade/mechanism exemption (governance §3b)")
        if in_prod:
            base = os.path.basename(rp)
            if re.search(r"_(hip|cuda|rocm)\.py$", base) and base != "__init__.py":
                r.fail("impl-naming", f"{rp}: platform word used as tier "
                       f"suffix (governance §3b: tiers are "
                       f"emu/triton/cpp/fork/ref or a vendor lib name)")

    gen = os.path.join(ROOT, "tools", "gen_tier_matrix.py")
    doc = os.path.join(ROOT, "docs", "tier-matrix.md")
    if not (os.path.isfile(gen) and os.path.isfile(doc)):
        r.fail("impl-naming", "tools/gen_tier_matrix.py or "
               "docs/tier-matrix.md missing")
        return
    try:
        proc = subprocess.run(
            [sys.executable, gen, "--check"],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            r.fail("impl-naming", proc.stderr.strip().splitlines()[-1]
                   if proc.stderr.strip() else "tier-matrix stale")
    except Exception as exc:  # generator crash is still a gate failure
        r.fail("impl-naming", f"tier-matrix check crashed: {exc}")


def check_dual_dist_packaging(r):
    """HARD: dual-dist layout invariants (governance §7.4).

    sgl-kernel (root pyproject) is a pure shadow kernel: no sitecustomize
    py-module, no entry-points, packages limited to sgl_kernel*. hcu-adapter
    (packaging/hcu-adapter) owns sitecustomize + the vllm entry-points and
    interlocks the kernel dist at build time: its setup.py/_pin.py renders
    sgl-kernel==<public>[+hcu.<stamp>] from the single kernel version source
    (sgl_kernel/version.py), and build.sh passes one DAS_BUILD_STAMP to both
    wheels. Without a stamp the pin names the public kernel version only.
    The adapter base dependencies also carry the validated torch/triton pins
    so a plain adapter install cannot drift to the newest nightly stack.
    Versions are independent lines; only the pin is machine-checked.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # py3.9/3.10 gate environments
        import tomli as tomllib

    root_pp = tomllib.loads((Path(ROOT) / "pyproject.toml").read_text(encoding="utf-8"))
    adapter_pp_path = Path(ROOT) / "packaging" / "hcu-adapter" / "pyproject.toml"
    adapter_pp = tomllib.loads(adapter_pp_path.read_text(encoding="utf-8"))

    kernel_project = root_pp["project"]
    if "entry-points" in kernel_project:
        r.fail("dual-dist", "root pyproject must not declare entry-points "
               "(they live in hcu-adapter)")
    if "sitecustomize" in root_pp.get("tool", {}).get("setuptools", {}).get(
            "py-modules", []):
        r.fail("dual-dist", "sitecustomize ships with hcu-adapter, not the "
               "kernel dist")
    include = root_pp.get("tool", {}).get("setuptools", {}).get(
        "packages", {}).get("find", {}).get("include")
    if include != ["sgl_kernel*"]:
        r.fail("dual-dist", f"kernel dist packages.find.include must be "
               f"['sgl_kernel*'], got {include!r}")

    version_src = (Path(ROOT) / "sgl_kernel" / "version.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__ = "(.+?)"$', version_src, re.M)
    if not m:
        r.fail("dual-dist", "sgl_kernel/version.py lacks __version__")
        return
    kernel_version = m.group(1)
    public_version = kernel_version.split("+", 1)[0]

    adapter_project = adapter_pp["project"]
    dynamic = adapter_project.get("dynamic") or []
    if "dependencies" not in dynamic or "version" not in dynamic:
        r.fail("dual-dist", "hcu-adapter version/dependencies must be dynamic "
               "(rendered by setup.py/_pin.py at build time)")
    try:
        import importlib.util
        pin_file = adapter_pp_path.parent / "_pin.py"
        spec = importlib.util.spec_from_file_location("hcu_adapter_build_pin", pin_file)
        pin_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pin_mod)
        pin = pin_mod.kernel_pin(stamp="")
    except Exception as exc:
        r.fail("dual-dist", f"cannot evaluate hcu-adapter interlock pin: {exc}")
    else:
        if pin != f"sgl-kernel=={public_version}":
            r.fail("dual-dist",
                   f"hcu-adapter interlock pin must be "
                   f"'sgl-kernel=={public_version}', got {pin!r}")
    if adapter_project.get("entry-points", {}).get("vllm.platform_plugins") != {
            "hcu_adapter": "hcu_adapter.bridges.vllm:hcu_platform_plugin"}:
        r.fail("dual-dist", "hcu-adapter vllm.platform_plugins entry-point "
               "drifted (zero-change migration rule)")
    if adapter_project.get("entry-points", {}).get("vllm.general_plugins") != {
            "hcu_adapter": "hcu_adapter.bridges.vllm:hcu_runtime_plugin"}:
        r.fail("dual-dist", "hcu-adapter vllm.general_plugins entry-point "
               "drifted (zero-change rule)")
    if "sitecustomize" not in adapter_pp.get("tool", {}).get(
            "setuptools", {}).get("py-modules", []):
        r.fail("dual-dist", "hcu-adapter must ship sitecustomize (py-modules)")

    # Docker layering split: hcu-runtime composes the subgroups by self-reference.
    extras = adapter_project.get("optional-dependencies") or {}
    runtime_groups = (
        "hcu-runtime-kernels",
        "hcu-runtime-serve",
        "hcu-runtime-aux",
    )
    missing_groups = [g for g in runtime_groups if g not in extras]
    if missing_groups:
        r.fail("dual-dist", f"hcu-runtime subgroups missing: {missing_groups}")
    else:
        expected = {f"hcu-adapter[{g}]" for g in runtime_groups}
        actual = set(extras.get("hcu-runtime") or [])
        if actual != expected:
            r.fail("dual-dist",
                   "hcu-runtime must compose hcu-runtime-kernels/serve/aux "
                   f"by self-reference; missing={sorted(expected - actual)} "
                   f"extra={sorted(actual - expected)}")

    adapter_version_file = adapter_pp_path.parent / "VERSION"
    if not adapter_version_file.read_text(encoding="utf-8").strip():
        r.fail("dual-dist", "packaging/hcu-adapter/VERSION is empty "
               "(adapter dist version single source)")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as failures")
    args = ap.parse_args()

    sys.path.insert(0, ROOT)
    r = Report()
    n_py = check_py_syntax(r)
    n_sh = check_shell_lf(r)
    n_sec = check_secrets(r)
    reg = registry_names()
    if not reg:
        r.fail("registry", "could not parse any HCU_* names from hcu_env.py")
    check_registry_docs(r, reg)
    check_catalog_refs(r, reg)
    check_routing_glue_imports(r)
    check_example_yaml(r)
    check_ops_import_depth(r)
    check_bare_environ(r, reg)
    check_glue_has_no_numerics(r)
    check_text_patch_allowlist(r)
    check_registry_ops_in_catalog(r)
    check_impl_naming(r)
    check_dual_dist_packaging(r)

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
