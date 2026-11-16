import importlib
import logging
from functools import lru_cache

import torch

from hcu_adapter.kill_switch import op_disabled

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _kernels():
    try:
        module = importlib.import_module("boltops.mqa_logits")
    except ModuleNotFoundError as error:
        if error.name not in ("boltops", "boltops.mqa_logits"):
            raise
        logger.warning("QSA MQA BoltOPs unavailable; using Torch reference")
        return None
    return module.tilelang_qsa_mqa_prefill, module.tilelang_qsa_mqa_decode


@lru_cache(maxsize=16)
def _supported_device(device):
    return torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0] == "gfx936"


def _supported(q, k):
    return (not op_disabled("qsa_mqa") and q.is_cuda and k.is_cuda
            and q.dtype == k.dtype == torch.bfloat16
            and q.ndim == 3 and q.shape[1:] == (4, 128)
            and _supported_device(q.device))


@lru_cache(maxsize=2)
def _log_success(kind):
    logger.info("QSA MQA BoltOPs %s active", kind)


def prefill(original, q, k, row_starts, row_ends, score_scale=None):
    if not _supported(q, k) or k.ndim != 3 or k.shape[1:] != (1, 128):
        return original(q, k, row_starts, row_ends, score_scale)
    kernels = _kernels()
    if kernels is None:
        return original(q, k, row_starts, row_ends, score_scale)
    with torch.cuda.device(q.device):
        result = kernels[0](q, k, row_starts, row_ends, score_scale,
                            effective_range=k.shape[0])
    _log_success("prefill")
    return result


def decode(original, q, k_cache, page_table, context_lens, max_model_len,
           score_scale=None, *, min_scored_len=0):
    if (not _supported(q, k_cache) or k_cache.ndim != 4
            or k_cache.shape[2:] != (1, 128)
            or k_cache.shape[1] <= 0 or 64 % k_cache.shape[1]):
        return original(q, k_cache, page_table, context_lens, max_model_len,
                        score_scale, min_scored_len=min_scored_len)
    kernels = _kernels()
    if kernels is None:
        return original(q, k_cache, page_table, context_lens, max_model_len,
                        score_scale, min_scored_len=min_scored_len)
    with torch.cuda.device(q.device):
        result = kernels[1](q, k_cache, page_table, context_lens,
                            max_model_len, score_scale)
    _log_success("decode")
    return result
