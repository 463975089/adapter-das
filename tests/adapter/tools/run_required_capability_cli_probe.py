"""Run the unmodified official CLI, then record its public Adapter reports.

The test-only probe distribution is installed exclusively in the review venv.
No production callback, discovery function or platform is replaced here.
Run from the isolated installed-wheel venv with HCU_REQUIRED_PROBE_EVENTS set:
python tests/adapter/tools/run_required_capability_cli_probe.py /new/result.json
Use HCU_ADAPTER_HOST=none and explicitly allocated HIP visibility for real probes;
importing this module never starts the CLI or writes evidence.
"""
import json
import os
from pathlib import Path
import runpy
import sys
import traceback


def main():
    output = Path(sys.argv[1])
    assert not output.exists()
    events = Path(os.environ['HCU_REQUIRED_PROBE_EVENTS'])
    assert not events.exists()
    sys.argv = ['vllm.entrypoints.openai.api_server', '--help']
    exit_code = 0
    error = None
    try:
        runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
        exit_code = 1
    finally:
        from hcu_adapter.bridges.vllm import extension_reports
        from hcu_adapter import hcu_env
        reports = [dict(entry_point=r.entry_point_name, name=r.extension_name,
                        status=r.status.value, capabilities=sorted(r.capabilities), reason=r.reason)
                   for r in extension_reports()]
        result = dict(exit_code=exit_code, error=error, reports=reports,
                      activation_events=[json.loads(line) for line in events.read_text().splitlines()]
                      if events.exists() else [],
                      required_extensions=hcu_env.resolve('HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS')[0],
                      required_capabilities=hcu_env.resolve('HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES')[0],
                      duplicate=os.environ.get('HCU_REQUIRED_PROBE_DUPLICATE', '0'),
                      scope='Installed real vLLM CLI/general-plugin startup before model loading; no model or worker evidence.')
        output.write_text(json.dumps(result, indent=2)+'\n')
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
