"""Thin official Worker subclass; runner/loader bodies remain upstream-owned."""

from vllm.config import set_current_vllm_config
from vllm.v1.worker.gpu_worker import Worker as OfficialWorker

from hcu_adapter.model_load import dispatch_post_model_load


class Worker(OfficialWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        with set_current_vllm_config(self.vllm_config):
            dispatch_post_model_load(self, load_dummy_weights=load_dummy_weights)
