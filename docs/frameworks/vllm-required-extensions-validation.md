# Startup gate validation on vLLM 0.32

This independent candidate is based on PR17
`35f8f0569df4ffbed045bf0dc18974eaafd5ab1a`. Runtime changes are limited to
passing startup requirements and rejecting duplicate capability owners. Its
wheel contains 194 Python files and excludes the separate FP8/lifecycle changes.

Source/installed wheel candidate: `14b6f33c18b5b3796d3b9aa01408f06013924db2`.
Adapter wheel SHA256:
`616f19676caabf09b1bddc58a3a2e901db32f69c6c194947f032e7fe740e3649`.
Target: vLLM `a198423b7be0ccf9c9ac3fb01c02ac03ab145e93`.
Subsequent changes add only this report and the reproducible test-only CLI probe.

The source phase and independently installed phase each passed **66 CPU tests**:
`test_vllm_extensions.py`, `test_vllm_bridge.py`, and
`test_vllm_required_startup.py`. These are the same tests in two phases, not 132
different tests. The installed phase clears PYTHONPATH, runs outside the source
tree with importlib test import mode, and checks source/wheel/installed bytes.
Tests include a real fork, but injected platform/discovery dependencies do not
themselves prove real vLLM startup.

The fresh installed candidate was therefore also exercised through unmodified
official CLI `--help`, with actual platform/general-plugin metadata discovery
on gfx936/DTK. The small test-only distribution under
`tests/adapter/fixtures/required_capability_probe` supplies two extension specs;
their activation writes an event file and installs no operator. The observer
under `tests/adapter/tools/run_required_capability_cli_probe.py` runs the official
module and records its public `extension_reports()`, without replacing discovery,
callbacks or platform behavior.

| Installed CLI case | Result | Observed reports / activation events |
| --- | --- | --- |
| Unique capability owners, required `upgrade-probe-left` | Exit 0 | Both active; 2 events carry the requested capability |
| Two different owners and priorities advertise `upgrade-probe-left` | Exit 1 | Both failed with duplicate-owner reason; **0 events** |
| Unique owners, required `upgrade-probe-missing` | Exit 1 | Required-capability error; unrelated specs activate and record 2 events |

The missing case deliberately shows that required failure does not roll back
unrelated successful activation. This is an admission gate, not an arbitrary
extension transaction. CLI help runs before model/worker construction; no
real-model, Graph/AOT, precision or performance acceptance is claimed for this
independent wheel. The broader integration matrix remains open.

To reproduce, build/install this Adapter and the test-only fixture in a **fresh
isolated environment** with the target vLLM wheel. Keep PYTHONPATH empty and use
the recorded real HCU platform selection/toolchain paths. Set
`VLLM_PLUGINS=hcu_adapter`, `HCU_ADAPTER_HOST=none` to defer activation to the
official plugins. For each case use fresh absolute output/event paths:

```bash
python -m pip wheel --no-deps --no-build-isolation \
  tests/adapter/fixtures/required_capability_probe -w /path/to/probe-wheels
python -m pip install --no-deps /path/to/probe-wheels/hcu_required_capability_validation_probe-0.1.0-py3-none-any.whl
export HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS=
export HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=upgrade-probe-left
export HCU_REQUIRED_PROBE_DUPLICATE=0
export HCU_REQUIRED_PROBE_EVENTS=/path/to/fresh-events.jsonl
python tests/adapter/tools/run_required_capability_cli_probe.py /path/to/fresh-report.json
```

For the duplicate case set `HCU_REQUIRED_PROBE_DUPLICATE=1`; for missing, use
unique owners and request `upgrade-probe-missing`. Both cases must exit 1;
inspect the JSON reports and events above. Do not install this test fixture in a
production/model environment.
