// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
/* HCU/DTK build shim — flashinfer's utils_hip.cuh includes
 * <cuda_device_runtime_api.h> (CUDA dynamic-parallelism device API).
 * Nothing in the sources we compile calls it on device; DTK's compat
 * copy has a deep conflicting closure. Intentionally empty: if some
 * future source actually needs a device-runtime symbol, the compile
 * will fail loudly here instead of silently mis-declaring.
 */
