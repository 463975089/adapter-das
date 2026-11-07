# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Text vocabulary and public reference API for composite HF checkpoints."""


def text_configuration(config):
    if not isinstance(config, dict):
        raise ValueError("checkpoint configuration must be a mapping")
    # Community get_hf_text_config prefers language_config over text_config.
    # DeepSeek-VL2 stores its decoder vocabulary and quantization here.
    for field in ("language_config", "text_config"):
        if field in config:
            text = config[field]
            if not isinstance(text, dict):
                raise ValueError(f"checkpoint {field} must be a mapping")
            return text
    return config


def vocabulary_size(config):
    size = text_configuration(config).get("vocab_size")
    if type(size) is not int or size <= 0:
        raise ValueError("checkpoint text vocabulary must be a positive integer")
    return size


def reference_loader(config, transformers):
    text = text_configuration(config)  # refuse malformed nested configurations
    if text is not config and isinstance(config.get("vision_config"), dict):
        return transformers.AutoModelForImageTextToText
    return transformers.AutoModelForCausalLM
