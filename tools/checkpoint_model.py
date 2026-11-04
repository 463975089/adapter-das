"""Text vocabulary and public reference API for composite HF checkpoints."""


def text_configuration(config):
    if not isinstance(config, dict):
        raise ValueError("checkpoint configuration must be a mapping")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("checkpoint text_config must be a mapping")
    return text


def vocabulary_size(config):
    size = text_configuration(config).get("vocab_size")
    if type(size) is not int or size <= 0:
        raise ValueError("checkpoint text vocabulary must be a positive integer")
    return size


def reference_loader(config, transformers):
    text_configuration(config)  # refuse malformed nested configurations
    if isinstance(config.get("text_config"), dict) and isinstance(config.get("vision_config"), dict):
        return transformers.AutoModelForImageTextToText
    return transformers.AutoModelForCausalLM
