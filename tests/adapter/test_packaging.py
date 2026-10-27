import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_wheel_contains_bootstrap_and_nested_adapter_packages(tmp_path):
    source = tmp_path / "source"
    wheel_dir = tmp_path / "wheel"
    shutil.copytree(
        REPO_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", "__pycache__", "build", "dist", "*.egg-info"
        ),
    )
    wheel_dir.mkdir()
    env = os.environ.copy()
    env["HCU_ADAPTER_HOST"] = "none"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(source),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    wheel = next(wheel_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())

    expected = {"sitecustomize.py"}
    for package in ("hcu_adapter", "sgl_kernel"):
        expected.update(
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / package).rglob("*.py")
        )

    assert not (expected - members), f"wheel is missing: {sorted(expected - members)}"
