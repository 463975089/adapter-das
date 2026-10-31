"""Thin official Worker subclass; runner/loader bodies remain upstream-owned."""

from vllm.config import set_current_vllm_config
from vllm.v1.worker.gpu_worker import Worker as OfficialWorker

from hcu_adapter.bridges.vllm_model_load import dispatch_model_mutation, dispatch_post_model_load


class Worker(OfficialWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        with set_current_vllm_config(self.vllm_config):
            dispatch_post_model_load(self, load_dummy_weights=load_dummy_weights)

    def reload_weights(self, *args, **kwargs) -> None:
        with set_current_vllm_config(self.vllm_config):
            dispatch_model_mutation(self, operation="reload_weights", role="target")
            return super().reload_weights(*args, **kwargs)

    def _start_weight_update(self, is_draft: bool = False) -> None:
        with set_current_vllm_config(self.vllm_config):
            dispatch_model_mutation(
                self, operation="start_weight_update",
                role="draft" if is_draft else "target",
            )
            return super()._start_weight_update(is_draft=is_draft)
