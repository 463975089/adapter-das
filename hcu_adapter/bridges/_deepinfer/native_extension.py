"""Explicitly expose separately built HIP native modules through the package seam."""

import json
from pathlib import Path

from hcu_adapter import registry


def install(artifacts):
    """Attach an isolated build directory without changing upstream files."""
    from deepinfer import __path__ as package_path

    root = Path(artifacts).resolve()
    manifest = json.loads((root / "hcu-build.json").read_text())
    if manifest["upstream_pin"] != "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01":
        raise RuntimeError("Native extension was built against a different upstream pin")
    provider = root / "deepinfer"
    if not list(provider.glob("_C*.so")):
        raise FileNotFoundError("Native extension artifact is absent: " + str(provider))
    # Put the explicit HIP artifact first, ahead of any NVIDIA binary directory.
    location = str(provider)
    if location not in package_path:
        package_path.insert(0, location)
    registry.register("deepinfer._C", "hcu_adapter.bridges._deepinfer.native_extension.install",
                      "deepinfer.__path__", "seam")
    print("HCU adapter runtime: deepinfer._C -> HIP staged build (candidate)")
