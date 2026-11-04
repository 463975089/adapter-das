# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import os

base = '/public/opendas/DL_DATA/llm-models'
candidates = [
    'qwen2.5/Qwen2.5-0.5B-Instruct',
    'qwen3/Qwen3-32B',
    'qwen3/Qwen3-30B-A3B',
    'Qwen3.5-35B-A3B',
    'qwen3.8/Qwen3.8-27B-FP8',
    'Qwen3.8-Flash-Next-FP8',
    'Qwen3.8-Flash-Next',
]
for name in candidates:
    p = os.path.join(base, name, 'config.json')
    if not os.path.exists(p):
        print(name, 'NO CONFIG')
        continue
    c = json.load(open(p))
    q = c.get('quantization_config') or {}
    print(name,
          '| quant=', q.get('quant_method', 'none'),
          '| fmt=', q.get('format', ''),
          '| arch=', (c.get('architectures') or ['?'])[0],
          '| layers=', c.get('num_hidden_layers'),
          '| hidden=', c.get('hidden_size'),
          '| experts=', c.get('num_experts', '-'))
