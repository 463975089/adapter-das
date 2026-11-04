"""Composite checkpoint vocabulary must come from the actual text decoder."""

from types import SimpleNamespace

import pytest

from tools.checkpoint_model import reference_loader, vocabulary_size
from tools.l15_model_smoke import validate_generation


def test_nested_vocabulary_admits_text_tokens_outside_outer_metadata():
    config = {"vocab_size": 16, "text_config": {"vocab_size": 128}, "vision_config": {}}
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
])
def test_invalid_text_config_refused_before_engine(config):
    with pytest.raises(ValueError, match="checkpoint"):
        vocabulary_size(config)


def test_reference_uses_public_conditional_generation_auto_class():
    api = SimpleNamespace(AutoModelForCausalLM=object(), AutoModelForImageTextToText=object())
    assert reference_loader({"vocab_size": 256}, api) is api.AutoModelForCausalLM
    assert reference_loader(
        {"text_config": {"vocab_size": 248320}, "vision_config": {}}, api
    ) is api.AutoModelForImageTextToText
    with pytest.raises(ValueError, match="text_config"):
        reference_loader({"text_config": None, "vision_config": {}}, api)
