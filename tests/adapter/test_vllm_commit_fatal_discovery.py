"""Real built fixture metadata and official plugin lifecycle; CPU only.

Run from an environment with this Adapter wheel's entry point installed.
Source phase may override Adapter with explicit PYTHONPATH. Installed phase
uses an outside-repo cwd and clears PYTHONPATH. Only the fixture target is added.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def installed_probe(tmp_path_factory):
    root = tmp_path_factory.mktemp("commit-probe-wheel")
    fixture = Path(__file__).parent / "fixtures/commit_fatal_probe"
    wheelhouse, target = root / "wheels", root / "site"
    wheelhouse.mkdir()
    # Build outside the source fixture: setuptools must not dirty the repo.
    import shutil
    source = root / "source"
    shutil.copytree(fixture, source)

    def pip(args):
        result = subprocess.run([sys.executable, "-B", "-m", "pip", args[0], "--no-index", *args[1:]],
                                cwd=root, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr

    pip(["wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheelhouse), str(source)])
    wheels = list(wheelhouse.glob("*.whl"))
    assert len(wheels) == 1
    pip(["install", "--no-deps", "--target", str(target), str(wheels[0])])
    return target


@pytest.mark.parametrize("spawn", [False, True])
@pytest.mark.parametrize("mode", [
    "clean", "fatal", "wrapped", "clean-badstr", "fatal-badstr",
    "wrapped-badstr", "factory-badstr", "runtime-badstr",
])
def test_actual_metadata_general_plugin_and_config_unpickle(
    installed_probe, tmp_path, mode, spawn,
):
    env = dict(os.environ, HCU_ADAPTER_HOST="none", VLLM_PLUGINS="hcu_adapter",
               CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1",
               HCU_COMMIT_PROBE_MODE=mode,
               HCU_COMMIT_PROBE_LOG=str(tmp_path / "events.jsonl"))
    env.pop("ROCR_VISIBLE_DEVICES", None)
    code = f"import site; site.addsitedir({str(installed_probe)!r}); from hcu_commit_fatal_probe import run; run({spawn!r})"
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                            cwd=tmp_path, text=True, capture_output=True, timeout=150)
    assert result.returncode == 0, result.stdout + result.stderr
    row = json.loads(result.stdout.split("COMMIT_PROBE=", 1)[1])
    assert not row["gpu_initialized"]
    assert row["config_roundtrip"] is spawn
    if spawn:
        assert row["start_method"] == "spawn" and row["child_exit"] == 0
    evidence = os.environ.get("HCU_COMMIT_PROBE_EVIDENCE")
    if evidence:
        destination = Path(evidence)
        destination.mkdir(exist_ok=True, parents=True)
        (destination / f"{mode}-spawn{int(spawn)}.json").write_text(
            json.dumps(row, indent=2) + "\n")


def test_fixture_import_has_no_log_or_runtime_patch_side_effects(installed_probe, tmp_path):
    env = dict(os.environ, HCU_ADAPTER_HOST="none", VLLM_PLUGINS="__disabled__",
               CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
               HCU_COMMIT_PROBE_ACTIVE="1", PYTHONDONTWRITEBYTECODE="1")
    for key in ("ROCR_VISIBLE_DEVICES", "HCU_COMMIT_PROBE_LOG", "HCU_COMMIT_PROBE_MODE"):
        env.pop(key, None)
    code = f'''
import site
site.addsitedir({str(installed_probe)!r})
import torch
names = ("init", "_lazy_init", "current_device", "get_device_properties")
before = [getattr(torch.cuda, name) for name in names]
import hcu_commit_fatal_probe
import hcu_commit_fatal_payload
assert before == [getattr(torch.cuda, name) for name in names]
assert not torch.cuda.is_initialized()
assert hcu_commit_fatal_payload.ProbeQuantizationConfig().get_name() == "cpu_commit_probe"
print("fixture import has no log/env requirement or runtime patch")
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env, cwd=tmp_path,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
