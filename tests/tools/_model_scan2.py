# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import os

base = '/public/opendas/DL_DATA/llm-models'
cands = [
    'vllm-awq-models/Qwen3-32B-AWQ',
    'vllm-fp8-models/Qwen3.5-122B-A10B-FP8',
    'vllm-w8a8-models/GLM-5-W8A8',
    'Qwen3.5-35B-A3B-W8A8',
    'Qwen3.5-122B-A10B-AWQ',
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
