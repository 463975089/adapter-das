# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import os

base = '/public/opendas/DL_DATA/llm-models'
cands = [
    # fp8 correct-format (quant_method=fp8)
    'qwen3.8/Qwen3.8-27B-FP8',
    'vllm-fp8-models/Qwen3.5-122B-A10B-FP8',
    'ZhipuAI/GLM-5-FP8',
    # int8 w8a8
    'glm5.3-channel-int8',
    'GLM-5.2-Channel-INT8-w8a8',
    'Qwen3.6-27B-Channel-INT8-w8a8',
    'Qwen3.5-4B-Channel-INT8-w8a8',
    # int4 w4a8
    'DeepSeek-V4.1-Flash-Channel-INT4-w4a8',
    'GLM-5.3-Flash-CHANNEL-INT4-w4a8-v3-same-as-iter_0033200',
    'GLM-5-Channel-INT4-w4a8',
]
for name in cands:
    for sub in ('', name + '/'):
        p = os.path.join(base, name, sub, 'config.json')
        if os.path.exists(p):
            break
    if not os.path.exists(p):
        # nested same-name dir (w4a8 v3 pattern)
        p = os.path.join(base, name, os.path.basename(name), 'config.json')
    if not os.path.exists(p):
        print(name, 'NO CONFIG')
        continue
    c = json.load(open(p))
    q = c.get('quantization_config') or {}
    arch = (c.get('architectures') or ['?'])[0]
    print(name,
          '| quant=', q.get('quant_method', 'none'),
          '| fmt=', q.get('format', ''),
          '| bits=', q.get('bits'),
          '| arch=', arch)
