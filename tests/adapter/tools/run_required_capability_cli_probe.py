"""Run the unmodified official CLI, then record its public Adapter reports.

The test-only probe distribution is installed exclusively in the review venv.
No production callback, discovery function or platform is replaced here.
"""
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

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
    reports = [dict(entry_point=r.entry_point_name, name=r.extension_name,
                    status=r.status.value, capabilities=sorted(r.capabilities), reason=r.reason)
               for r in extension_reports()]
    result = dict(exit_code=exit_code, error=error, reports=reports,
                  activation_events=[json.loads(line) for line in events.read_text().splitlines()]
                  if events.exists() else [],
                  required_extensions=os.environ.get('HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS', ''),
                  required_capabilities=os.environ.get('HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES', ''),
                  duplicate=os.environ.get('HCU_REQUIRED_PROBE_DUPLICATE', '0'),
                  scope='Installed real vLLM CLI/general-plugin startup before model loading; no model or worker evidence.')
    output.write_text(json.dumps(result, indent=2)+'\n')
sys.exit(exit_code)
