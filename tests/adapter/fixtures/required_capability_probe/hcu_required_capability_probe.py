"""Test-only installed providers used to observe real startup owner admission."""
import json
import os
from pathlib import Path

from hcu_adapter.extensions import ExtensionSpec


def _spec(name, capability, priority):
    def activate(context):
        event = dict(owner=name, capabilities=sorted(context.requested_capabilities),
                     extensions=sorted(context.requested_extensions))
        with Path(os.environ['HCU_REQUIRED_PROBE_EVENTS']).open('a') as stream:
            stream.write(json.dumps(event)+'\n')
    return ExtensionSpec(name, 1, priority, {capability}, activate)


def left():
    return _spec('probe-left-owner', 'upgrade-probe-left', 100)


def right():
    capability = ('upgrade-probe-left' if os.environ.get('HCU_REQUIRED_PROBE_DUPLICATE') == '1'
                  else 'upgrade-probe-right')
    return _spec('probe-right-owner', capability, 1)
