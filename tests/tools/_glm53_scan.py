# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json

p = ("/models/GLM-5.3-Flash-CHANNEL-INT4-w4a8-v3-same-as-iter_0033200/"
     "GLM-5.3-Flash-CHANNEL-INT4-w4a8-v3-same-as-iter_0033200/config.json")
c = json.load(open(p))
q = c.get('quantization_config') or {}
print('arch:', c.get('architectures'))
print('quant:', q.get('quant_method'), '| fmt:', q.get('format'),
      '| bits:', q.get('bits'))
print('layers:', c.get('num_hidden_layers'), '| hidden:', c.get('hidden_size'))
print('experts:', c.get('num_experts'), '| index_kpool:',
      c.get('index_kpool'), '| linear:', c.get('linear_attn_config') is not None)
