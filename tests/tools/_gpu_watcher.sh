#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:$LD_LIBRARY_PATH
echo "=== memory use ==="
/opt/dtk/bin/rocm-smi --showmemuse 2>/dev/null | grep -F 'HCU['
echo "=== compute use ==="
/opt/dtk/bin/rocm-smi --showuse 2>/dev/null | grep -F 'HCU['
echo "=== processes on cards ==="
/opt/dtk/bin/rocm-smi --showpids 2>/dev/null | tail -25
