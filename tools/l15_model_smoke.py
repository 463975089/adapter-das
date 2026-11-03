"""Single-HCU real Engine startup/generation, dummy and checkpoint evidence split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from tools.l15_runtime import audit_source, preflight, run_job, test_env, write_json


def tiny_config(family):
    config = dict(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, rms_norm_eps=1e-6,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        tie_word_embeddings=False, torch_dtype="bfloat16",
    )
    if family == "dense":
        config.update(model_type="llama", architectures=["LlamaForCausalLM"])
    elif family == "moe":
        config.update(
            model_type="qwen3_moe", architectures=["Qwen3MoeForCausalLM"],
            hidden_size=256, intermediate_size=512, head_dim=64,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128,
            norm_topk_prob=True, decoder_sparse_step=1,
        )
    elif family == "mla":
        config.update(
            model_type="deepseek_v2", architectures=["DeepseekV2ForCausalLM"],
            q_lora_rank=64, kv_lora_rank=512, qk_rope_head_dim=64,
            qk_nope_head_dim=128, v_head_dim=128, n_routed_experts=4,
            n_shared_experts=1, num_experts_per_tok=2, moe_intermediate_size=128,
            first_k_dense_replace=0, topk_method="greedy", n_group=1,
            topk_group=1, norm_topk_prob=True, routed_scaling_factor=1.0,
            moe_layer_freq=1, scoring_func="softmax",
        )
    elif family == "mamba":
        config.update(
            model_type="mamba2", architectures=["Mamba2ForCausalLM"],
            expand=2, num_heads=8, head_dim=32, state_size=16,
            n_groups=1, conv_kernel=4, layer_norm_epsilon=1e-5,
        )
    elif family == "hybrid":
        config.update(
            model_type="qwen3_next", architectures=["Qwen3NextForCausalLM"],
            head_dim=32, linear_key_head_dim=32, linear_value_head_dim=32,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2,
            moe_intermediate_size=128, shared_expert_intermediate_size=128,
            layer_types=["linear_attention", "full_attention"],
            full_attention_interval=2,
        )
    else:
        raise ValueError(f"unknown tiny model family: {family}")
    return config


def validate_generation(document, vocab_size, expected_count=8):
    items = document if isinstance(document, list) else [document]
    if len(items) != 2:
        raise ValueError("both requests must produce a generation")
    for item in items:
        ids = item.get("output_ids", [])
        if len(ids) != expected_count or any(
            type(token) is not int or not 0 <= token < vocab_size for token in ids
        ):
            raise ValueError("generation has missing/out-of-vocabulary tokens")
        probabilities = item.get("meta_info", {}).get("output_token_logprobs", [])
        if len(probabilities) != expected_count or any(
            not math.isfinite(entry[0]) or entry[1] != token
            for entry, token in zip(probabilities, ids)
        ):
            raise ValueError("generation has missing/non-finite token logprobs")


def validate_mode(info, mode):
    states = info.get("internal_states", [])
    if not states:
        raise ValueError("missing scheduler runtime state")
    for state in states:
        sizes = state.get("decode_cuda_graph_bs")
        memory = state.get("memory_usage", {}).get("graph", {}).get("decode")
        if not isinstance(memory, (int, float)) or not math.isfinite(memory) or memory < 0:
            raise ValueError("missing actual decode graph allocation evidence")
        # v0.5.22 startup installs the runner directly and may leave the
        # role-switch-only capture-bs bookkeeping empty. Captured decode
        # memory is independently reported by the actual runner.
        if mode == "graph" and memory <= 0:
            raise ValueError("missing actual decode graph allocation evidence")
        if mode == "eager" and (sizes or memory):
            raise ValueError("eager smoke unexpectedly captured decode graphs")


def worker(args):
    import sglang as sgl

    config = json.loads((args.model / "config.json").read_text())
    options = dict(
        model_path=str(args.model), dtype="bfloat16", random_seed=42,
        tp_size=1, attention_backend="triton", max_total_tokens=512,
        context_length=128, max_running_requests=4, mem_fraction_static=0.25,
        cuda_graph_max_bs_decode=2, disable_cuda_graph=args.mode == "eager",
    )
    if args.family != "checkpoint":
        options.update(load_format="dummy", skip_tokenizer_init=True)
    with sgl.Engine(**options) as engine:
        generations = []
        # Two batches test fresh inputs and cache reuse through the real
        # tokenizer/scheduler/prefill/decode path, rather than a model mock.
        for wave in range(2):
            generation_args = dict(
                sampling_params=dict(temperature=0, max_new_tokens=8, ignore_eos=True),
                return_logprob=True, logprob_start_len=0,
            )
            if args.family == "checkpoint":
                generation_args["prompt"] = (
                    ["The capital of France is", "2 + 2 ="] if wave == 0 else
                    ["The capital of China is", "3 + 3 ="]
                )
            else:
                generation_args["input_ids"] = [
                    [1, 7 + wave, 12, 9], [1, 18, 17 + wave, 6, 5],
                ]
            result = engine.generate(**generation_args)
            validate_generation(result, config["vocab_size"])
            generations.append(result)
        info = engine.get_server_info()
        validate_mode(info, args.mode)
    write_json(args.out, {
        "artifact": "l15-model-worker", "passed": True, "family": args.family,
        "mode": args.mode, "model": str(args.model),
        "weight_evidence": "real-checkpoint" if args.family == "checkpoint" else "dummy",
        "generations": generations, "server_info": info,
    })


def run(args):
    root = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    work = output.parent / (output.stem + "-jobs")
    work.mkdir(parents=True, exist_ok=True)
    families = args.families.split(",")
    modes = args.modes.split(",")
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"eager", "graph"}:
        raise ValueError("select distinct eager/graph modes")
    if not families or len(set(families)) != len(families):
        raise ValueError("select distinct model families")
    for family in families:
        if family == "checkpoint":
            if args.checkpoint is None:
                raise ValueError("checkpoint family requires --checkpoint")
            if not (args.checkpoint / "config.json").is_file() or not any(
                args.checkpoint.glob("*.safetensors")
            ):
                raise ValueError("checkpoint must be a local config and safetensors weights")
        else:
            tiny_config(family)
    runtime = preflight(args.up_tree.resolve(), root, args.binding_manifest.resolve())
    jobs = []
    for family in families:
        if family == "checkpoint":
            model = args.checkpoint.resolve()
        else:
            model = work / ("tiny-" + family)
            write_json(model / "config.json", tiny_config(family))
        for mode in modes:
            report = work / f"{family}-{mode}.json"
            report.unlink(missing_ok=True)
            job = run_job(
                [sys.executable, "-m", "tools.l15_model_smoke", "--worker",
                 "--model", str(model), "--family", family, "--mode", mode,
                 "--out", str(report)],
                cwd=root, env=test_env(args.up_tree.resolve(), root),
                timeout=args.timeout, log=work / f"{family}-{mode}.log",
            )
            try:
                evidence = json.loads(report.read_text())
            except (OSError, ValueError):
                evidence = {}
            job.update(family=family, mode=mode, evidence=evidence)
            job["passed"] = (
                job["returncode"] == 0 and not job["timed_out"]
                and evidence.get("artifact") == "l15-model-worker"
                and evidence.get("passed") is True
                and evidence.get("mode") == mode
                and evidence.get("family") == family
            )
            integrity = audit_source(args.up_tree.resolve(), runtime["upstream"]["commit"])
            job["source_integrity_after"] = integrity
            job["passed"] = job["passed"] and integrity["passed"]
            jobs.append(job)
            write_json(output, {
                "artifact": "l15-model-smoke", "runtime": runtime,
                "passed": len(jobs) == len(families) * len(modes) and all(j["passed"] for j in jobs),
                "complete": len(jobs) == len(families) * len(modes),
                "families": families, "modes": modes, "jobs": jobs,
            })
            print(f"model smoke: {family}/{mode} {'PASS' if job['passed'] else 'FAIL'}", flush=True)
            if not integrity["passed"]:
                return False  # do not launch further models on mutated source
    return all(job["passed"] for job in jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--up-tree", type=Path)
    parser.add_argument("--families", default="dense,moe,mla,mamba,hybrid,checkpoint")
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--binding-manifest", type=Path,
                        default=Path("tests/manifests/l15_v0522.json"))
    parser.add_argument("--out", type=Path, default=Path("results/l15-readiness/models.json"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--family", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("eager", "graph"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0
    try:
        if args.up_tree is None or args.timeout <= 0:
            raise ValueError("--up-tree and positive --timeout are required")
        return 0 if run(args) else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        write_json(args.out, {"artifact": "l15-model-smoke", "passed": False,
                             "complete": False, "error": str(exc), "jobs": [],
                             "source_integrity": getattr(exc, "report", None)})
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
