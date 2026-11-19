# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Thin official Worker subclass; runner/loader bodies remain upstream-owned."""

from vllm.config import set_current_vllm_config
from vllm.v1.worker.gpu_worker import Worker as OfficialWorker

from hcu_adapter.bridges.vllm_mhc import check_graph_providers, prepare_weight_mutation
from hcu_adapter.bridges.vllm_model_load import dispatch_model_mutation, dispatch_post_model_load

from hcu_adapter.features.dsv41.v41_fp8_indexer import check_graph_fp4_provider


def _check_optimized_owners():
    from hcu_adapter.bridges.vllm import (
        extension_reports, require_extension_healthy, require_runtime_healthy,
    )
    from hcu_adapter.extensions import ExtensionStatus

    require_runtime_healthy()
    active = {capability for report in extension_reports()
              if report.status is ExtensionStatus.ACTIVE
              for capability in report.capabilities}
    for capability in ("hcu-v41-fp8-indexer-eager-reference", "dsv41-wo-a-token-major"):
        if capability in active:
            require_extension_healthy(capability)


class Worker(OfficialWorker):
    def _make_model_runner(self):
        from hcu_adapter.bridges.vllm_model_runner import create_model_runner

        return create_model_runner(
            self.vllm_config, self.device, default=super()._make_model_runner,
        )

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with set_current_vllm_config(self.vllm_config):
            _check_optimized_owners()
        super().load_model(load_dummy_weights=load_dummy_weights)
        with set_current_vllm_config(self.vllm_config):
            _check_optimized_owners()
            dispatch_post_model_load(self, load_dummy_weights=load_dummy_weights)
            from hcu_adapter.bridges.vllm_flashmla_decode import qualify_after_model_load

            qualify_after_model_load(self.vllm_config, self.device)

    def _check_graph_dispatch(self) -> None:
        if not self.vllm_config.model_config.enforce_eager:
            check_graph_providers()
            check_graph_fp4_provider()

    def execute_model(self, scheduler_output):
        self._check_graph_dispatch()
        return super().execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):
        self._check_graph_dispatch()
        return super().sample_tokens(grammar_output)

    def execute_dummy_batch(self) -> None:
        self._check_graph_dispatch()
        super().execute_dummy_batch()

    def reload_weights(self, *args, **kwargs) -> None:
        with set_current_vllm_config(self.vllm_config):
            _check_optimized_owners()
            dispatch_model_mutation(self, operation="reload_weights", role="target")
            prepare_weight_mutation(self.vllm_config)
            return super().reload_weights(*args, **kwargs)

    def _start_weight_update(self, is_draft: bool = False) -> None:
        with set_current_vllm_config(self.vllm_config):
            _check_optimized_owners()
            dispatch_model_mutation(
                self, operation="start_weight_update",
                role="draft" if is_draft else "target",
            )
            prepare_weight_mutation(self.vllm_config)
            return super()._start_weight_update(is_draft=is_draft)
