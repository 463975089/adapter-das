# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import os

base = '/public/opendas/DL_DATA/llm-models/qwen3.5'
for name in ('Qwen3.5-35B-A3B-W8A8', 'Qwen3.5-122B-A10B-AWQ'):
    p = os.path.join(base, name, 'config.json')
    if not os.path.exists(p):
        print(name, 'NO CONFIG; files:',
              sorted(os.listdir(os.path.join(base, name)))[:8])
        continue
    c = json.load(open(p))
    q = c.get('quantization_config') or {}
    print(name,
          '| quant=', q.get('quant_method', 'none'),
          '| bits=', q.get('bits'),
          '| fmt=', q.get('format', ''),
          '| arch=', (c.get('architectures') or ['?'])[0],
          '| experts=', c.get('num_experts', '-'))
