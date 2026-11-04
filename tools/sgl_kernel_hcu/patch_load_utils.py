"""Patch the aot python wrapper's op loader for our top-level .so layout.

The stock `_load_architecture_specific_ops` in
python/sgl_kernel/load_utils.py only globs `sm90/common_ops.*` /
`sm100/common_ops.*` subdirectories. setup_hcu.py builds the extension at
the package top level (where setuptools puts `sgl_kernel.common_ops`), and
no sm90/sm100 payloads are shipped — the exact combination that produced
the empty-op wheels in the dtk2604 images (torch.ops.sgl_kernel == [] at
runtime, verified 2026-11-03).

This patch points the glob at the package top level. Run against the BUILD
WORKSPACE copy only; the official tree stays untouched.
"""

import sys
from pathlib import Path

OLD = '    ops_pattern = str(sgl_kernel_dir / ops_subdir / "common_ops.*")'
NEW = (
    '    ops_pattern = str(sgl_kernel_dir / "common_ops.*")  '
    "# HCU build: .so lives at package top level"
)


def main(path: str) -> int:
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    if NEW.split(" # ")[0] in src:
        print(f"patch_load_utils: already patched: {p}")
        return 0
    if OLD not in src:
        print(
            f"patch_load_utils: anchor not found in {p} — upstream layout "
            "changed; inspect load_utils._load_architecture_specific_ops "
            "and update the patch."
        )
        return 1
    p.write_text(src.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patch_load_utils: patched {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
