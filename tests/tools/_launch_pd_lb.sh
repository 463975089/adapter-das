#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# PD disaggregation - mini load balancer (sglang_router, CPU only)
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
mkdir -p /root/logs
exec python -m sglang_router.launch_router \
  --pd-disaggregation \
  --mini-lb \
  --prefill http://127.0.0.1:30190 \
  --decode http://127.0.0.1:30191 \
  --host 0.0.0.0 --port 30090 \
  > /root/logs/pd_lb_30090.log 2>&1
