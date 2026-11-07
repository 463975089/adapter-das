"""CPU source-contract audit of DeepSeek's DeepEP routed scaling.

Execute the exact upstream method body with explicit CPU gate/TopK/expert
doubles. This audits caller composition, not the real host, kernels,
dispatch/combine, checkpoint weights or model accuracy.
"""

import argparse
import ast
import copy
import hashlib
import json
import itertools
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import torch

RELATIVE_SOURCE = "python/sglang/srt/models/deepseek_v2.py"


def _isolated_cpu():
    from hcu_adapter.hcu_env import resolve

    if (resolve("HCU_ADAPTER_HOST")[0] != "none"
            or any(os.environ.get(key) != "" for key in (
                "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
            )) or torch.cuda.is_initialized()):
        raise RuntimeError("Run in a fresh HOSTnone process with all GPU visibility empty")


def _load_method(raw, filename):
    tree = ast.parse(raw, filename=str(filename))
    owner = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV2MoE"
    )
    method = copy.deepcopy(next(
        node for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_deepep"
    ))
    if method.decorator_list:
        raise ValueError("Decorated caller requires a separate source-contract audit")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations"),
        ], level=0), method,
    ], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "_use_aiter": False,
        "envs": NS(SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO=NS(
            get=lambda: False)),
    }
    exec(compile(module, str(filename), "exec"), namespace)
    return namespace["forward_deepep"], namespace


def _exercise(function, namespace):
    rows = []
    cases = itertools.product((1., 1.5, .25), *[(False, True)] * 5)
    for factor, shared, fused, aiter, empty, nextn in cases:
        namespace["_use_aiter"] = aiter
        tokens = 0 if empty else 3
        value = torch.ones(tokens, 8, dtype=torch.bfloat16)
        routed = torch.arange(tokens * 8).reshape(tokens, 8).to(
            torch.bfloat16) / 8
        shared_value = torch.full_like(value, .0625)
        marker, calls = object(), []
        location = object()
        namespace["ExpertLocationDispatchInfo"] = NS(init_new=lambda **kwargs: location)

        class TopK:
            def __call__(self, *args, **kwargs):
                assert kwargs["expert_location_dispatch_info"] is (
                    None if nextn else location
                )
                calls.append("topk")
                return marker

            def empty_topk_output(self, *args, **kwargs):
                calls.append("empty-topk")
                return marker

        class Experts:
            should_fuse_routed_scaling_factor_in_topk = fused

            def __call__(self, *, hidden_states, topk_output):
                assert hidden_states is value and topk_output is marker
                calls.append("experts")
                # Explicit double: represent weights that have
                # already been scaled in either TopK branch.
                return routed * factor if fused or aiter else routed.clone()

        def gate(*args, **kwargs):
            calls.append("gate")
            return torch.zeros(tokens, 2)

        def shared_experts(*args):
            calls.append("shared")
            return shared_value.clone() if shared else None

        model = NS(
            _fuse_shared_experts_inside_sbo=False,
            is_nextn=nextn, is_hash=False, layer_id=0,
            num_fused_shared_experts=0, alt_stream=None,
            gate=gate, topk=TopK(), experts=Experts(),
            routed_scaling_factor=factor,
            _forward_shared_experts=shared_experts,
        )
        batch = NS(moe_num_token_non_padded=lambda: tokens)
        result = function(model, value, batch)
        expected = routed.float() * factor
        if shared and not empty:
            expected += shared_value.float()
        torch.testing.assert_close(
            result, expected.bfloat16(), rtol=0, atol=0,
            msg="Caller must apply routed scaling exactly once",
        )
        assert torch.equal(value, torch.ones_like(value))
        assert calls == (
            ["empty-topk", "experts"] if empty else
            ["gate", "shared", "topk", "experts"]
        ), calls
        rows.append(dict(
            factor=factor, shared=shared, fused_topk=fused,
            aiter_topk=aiter, empty=empty, nextn=nextn, exact=True,
        ))
    return rows


def audit(up_tree):
    _isolated_cpu()
    up_tree = Path(up_tree).resolve()
    source = up_tree / RELATIVE_SOURCE
    raw = source.read_bytes()
    git = ["git", "-c", f"safe.directory={up_tree}", "-C", str(up_tree)]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(git + ["show", f"{commit}:{RELATIVE_SOURCE}"]) != raw:
        raise ValueError("Caller source differs from its bound Git commit")
    function, namespace = _load_method(raw, source)
    checks = _exercise(function, namespace)
    assert raw == source.read_bytes(), "Caller source changed during audit"
    assert commit == subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    assert not torch.cuda.is_initialized()
    return dict(
        source_contract_passed=True, commit=commit,
        source_path=RELATIVE_SOURCE, source_sha256=hashlib.sha256(raw).hexdigest(),
        checks=checks, gpu_initialized=False,
        scope="Exact source method with CPU doubles; ordinary LL path, no overlap",
        real_host_acceptance=False, distributed_acceptance=False, model_acceptance=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--up-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.up_tree)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"CPU caller source contract: {len(report['checks'])} checks passed; "
          "real host/distributed/model acceptance pending")


if __name__ == "__main__":
    main()
