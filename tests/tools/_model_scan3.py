# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import os

base = '/public/opendas/DL_DATA/llm-models'
cands = [
    'Qwen3.5-35B-A3B-W8A8',
    'Qwen3.5-122B-A10B-AWQ',
    'Qwen3.5-27B-W8A8',
    'Qwen3.5-4B-Channel-INT8-w8a8',
]
for name in cands:
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
          '| experts=', c.get('num_experts', '-'))
# AWQ dir has no config at top: check one level down
d = os.path.join(base, 'vllm-awq-models/Qwen3-32B-AWQ')
for root, dirs, files in os.walk(d):
    for f in files:
        if f == 'config.json':
            c = json.load(open(os.path.join(root, f)))
            q = c.get('quantization_config') or {}
            print('AWQ found at', os.path.relpath(root, d),
                  '| quant=', q.get('quant_method', 'none'),
                  '| bits=', q.get('bits'),
                  '| arch=', (c.get('architectures') or ['?'])[0])
    if root != d:
        break
