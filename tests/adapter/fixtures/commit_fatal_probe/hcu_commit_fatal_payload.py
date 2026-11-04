# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""A real quant-config pickle payload; never creates weights or a model."""

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from hcu_commit_fatal_probe import _cpu_guard, emit


class ProbeQuantizationConfig(QuantizationConfig):
    def __setstate__(self, state):
        # Called by real spawn unpickling, never by importing this module.
        _cpu_guard()
        self.__dict__.update(state)
        emit("payload-unpickle")

    def get_name(self):
        return "cpu_commit_probe"

    def get_supported_act_dtypes(self):
        return []

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        raise AssertionError("CPU pickle probe must not construct quantized layers")
