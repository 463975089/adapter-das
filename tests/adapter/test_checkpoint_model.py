# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Composite checkpoint vocabulary must come from the actual text decoder."""

from types import SimpleNamespace

import pytest

from tools.checkpoint_model import reference_loader, text_configuration, vocabulary_size
from tools.l15_model_smoke import validate_generation


@pytest.mark.parametrize("field", ["text_config", "language_config"])
def test_nested_vocabulary_admits_text_tokens_outside_outer_metadata(field):
    config = {"vocab_size": 16, field: {"vocab_size": 128}, "vision_config": {}}
    result = [{"output_ids": [127] * 8,
               "meta_info": {"output_token_logprobs": [[-.1, 127, None]] * 8}}
              for _ in range(2)]
    validate_generation(result, vocabulary_size(config))
    result[0]["output_ids"][0] = 128
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        validate_generation(result, vocabulary_size(config))
    assert vocabulary_size({"vocab_size": 256}) == 256
    assert vocabulary_size({"text_config": {"vocab_size": 248320}}) == 248320


@pytest.mark.parametrize("config", [
    {}, {"vocab_size": True}, {"vocab_size": -1}, {"vocab_size": "128"},
    {"vocab_size": 256, "text_config": None},
    {"vocab_size": 256, "text_config": {"vocab_size": 0}},
    {"vocab_size": 256, "language_config": None},
    {"vocab_size": 256, "language_config": {"vocab_size": 0}},
    {"vocab_size": 256, "language_config": []},
])
def test_invalid_text_config_refused_before_engine(config):
    with pytest.raises(ValueError, match="checkpoint"):
        vocabulary_size(config)


@pytest.mark.parametrize("field", ["text_config", "language_config"])
def test_reference_uses_public_conditional_generation_auto_class(field):
    api = SimpleNamespace(AutoModelForCausalLM=object(), AutoModelForImageTextToText=object())
    assert reference_loader({"vocab_size": 256}, api) is api.AutoModelForCausalLM
    assert reference_loader(
        {field: {"vocab_size": 248320}, "vision_config": {}}, api
    ) is api.AutoModelForImageTextToText
    with pytest.raises(ValueError, match="text_config"):
        reference_loader({"text_config": None, "vision_config": {}}, api)


def test_language_config_precedes_text_metadata_and_preserves_quantization():
    language = {"vocab_size": 129280, "model_type": "deepseek_v2",
                "quantization_config": {"quant_method": "compressed-tensors"}}
    config = {"vocab_size": 16, "text_config": {"vocab_size": 32},
              "language_config": language, "vision_config": {}}
    assert text_configuration(config) is language
    assert vocabulary_size(config) == 129280
    assert text_configuration(config)["quantization_config"] == language["quantization_config"]


def test_invalid_language_config_does_not_fall_back_to_text_metadata():
    config = {"language_config": None, "text_config": {"vocab_size": 128}}
    with pytest.raises(ValueError, match="language_config"):
        vocabulary_size(config)
