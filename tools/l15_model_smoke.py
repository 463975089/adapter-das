# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU real Engine startup/generation, dummy and checkpoint evidence split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from tools.l15_runtime import audit_source, preflight, run_job, test_env, write_json
from tools.checkpoint_files import audit as audit_checkpoint, fingerprint
from tools.checkpoint_model import text_configuration, vocabulary_size


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINDING_MANIFEST = REPO_ROOT / "tests/manifests/l15_v0522.json"


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


def probe_cache_reuse(engine, request, vocab_size):
    """Request only output logprobs so prompt logprobs do not force recompute."""
    import copy

    request = copy.deepcopy(request)
    request["logprob_start_len"] = -1
    warm = engine.generate(**request)
    replay = engine.generate(**request)
    validate_generation(warm, vocab_size)
    validate_generation(replay, vocab_size)
    counts = [item.get("meta_info", {}).get("cached_tokens") for item in replay]
    if any(type(count) is not int or count <= 0 for count in counts):
        raise ValueError("cache probe did not observe cached tokens for every replay request")
    if any(a["output_ids"] != b["output_ids"] for a, b in zip(warm, replay)):
        raise ValueError("cache replay changed greedy tokens")
    return {"passed": True, "cached_tokens": counts, "cache_hit_proven": True}


def validate_request_replay(generations):
    """Require deterministic greedy tokens after an intervening request batch.

    Logprob differences are recorded, not used as a substitute for independent
    model accuracy. Scheduler/cache-dependent arithmetic may change rounding.
    """
    if len(generations) != 3:
        raise ValueError("request replay requires original, changed, replay batches")
    maximum = 0.0
    for original, replay in zip(generations[0], generations[2]):
        if original["output_ids"] != replay["output_ids"]:
            raise ValueError("replayed request tokens changed after intervening inputs")
        before = original["meta_info"]["output_token_logprobs"]
        after = replay["meta_info"]["output_token_logprobs"]
        maximum = max(maximum, *(abs(a[0] - b[0]) for a, b in zip(before, after)))
    return {"passed": True, "greedy_tokens_equal": True,
            "max_logprob_difference": maximum,
            "cache_hit_proven": False}


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


def validate_lm_head_option(info, requested):
    if info.get("enable_fp32_lm_head", False) is not requested:
        raise ValueError("runtime FP32 LM-head option differs from requested configuration")


def validate_execution_dtype(info, requested):
    if requested not in ("bfloat16", "float32") or info.get("dtype", "bfloat16") != requested:
        raise ValueError("runtime dtype differs from requested configuration")


def validate_quantization_option(info, requested):
    if info.get("quantization") != requested:
        raise ValueError("runtime quantization option differs from requested configuration")


def validate_memory_fraction(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("memory fraction must be finite and strictly between zero and one")


def validate_memory_option(info, requested):
    validate_memory_fraction(requested)
    actual = info.get("mem_fraction_static", .25)
    validate_memory_fraction(actual)
    if actual != requested:
        raise ValueError("runtime memory fraction differs from requested configuration")


def validate_tp_size(value):
    if type(value) is not int or value not in (1, 2, 4, 8):
        raise ValueError("select a TP size of 1, 2, 4 or 8")


def validate_tp_option(info, requested):
    validate_tp_size(requested)
    actual = info.get("tp_size", 1)
    if type(actual) is not int or actual != requested:
        raise ValueError("runtime TP size differs from requested configuration")
    for state in info.get("internal_states", []):
        actual = state.get("tp_size", 1)
        if type(actual) is not int or actual != requested:
            raise ValueError("scheduler TP size differs from requested configuration")


def worker(args):
    tp_size = getattr(args, "tp_size", 1)
    validate_tp_size(tp_size)
    mem_fraction = getattr(args, "mem_fraction_static", .25)
    validate_memory_fraction(mem_fraction)
    checkpoint = None
    if args.family == "checkpoint":
        if args.checkpoint_manifest is None:
            raise ValueError("checkpoint worker requires controller byte manifest")
        checkpoint = json.loads(args.checkpoint_manifest.read_text())
        if not audit_checkpoint(args.model, checkpoint)["passed"]:
            raise ValueError("checkpoint changed before model loading")
    import sglang as sgl

    config = json.loads((args.model / "config.json").read_text())
    vocab_size = vocabulary_size(config)
    profile = getattr(args, "profile", "smoke")
    fp32_head = getattr(args, "enable_fp32_lm_head", False)
    dtype = getattr(args, "dtype", "bfloat16")
    quantization = getattr(args, "quantization", None)
    accuracy_cases = None
    if profile == "accuracy":
        if args.family != "checkpoint":
            raise ValueError("accuracy profile requires a real checkpoint")
        from transformers import AutoTokenizer
        from tools.checkpoint_accuracy import cases, validate_cases

        accuracy_cases = cases(AutoTokenizer.from_pretrained(str(args.model), local_files_only=True))
        validate_cases(accuracy_cases)
    options = dict(
        model_path=str(args.model), dtype=dtype, random_seed=42,
        tp_size=tp_size, attention_backend="triton", max_total_tokens=512,
        context_length=128, max_running_requests=4, mem_fraction_static=mem_fraction,
        cuda_graph_max_bs_decode=2, disable_cuda_graph=args.mode == "eager",
    )
    if args.family != "checkpoint":
        options.update(load_format="dummy", skip_tokenizer_init=True)
    if accuracy_cases is not None:
        options.update(context_length=4096, max_total_tokens=16384)
    if fp32_head:
        options["enable_fp32_lm_head"] = True
    if quantization is not None:
        options["quantization"] = quantization
    with sgl.Engine(**options) as engine:
        if getattr(args, "host", "adapter") == "adapter+plugin":
            # load_plugins() swallows plugin failures into a log line; a
            # silent non-load here would read as a green stacked run.
            from sglang.srt.plugins.hook_registry import HookRegistry

            sources = {src.plugin_name for hooks in HookRegistry._hooks.values()
                       for _, _, src in hooks if src is not None}
            if "hcu" not in sources:
                raise ValueError(
                    "hcu-sglang-plugin did not load in the engine process "
                    f"(sources={sorted(sources)})"
                )
        generations = []
        # Original -> changed -> original tests request isolation. Actual cache
        # hits need separate runtime counters and are not inferred from equality.
        for wave in range(3):
            input_wave = 0 if wave == 2 else wave
            generation_args = dict(
                sampling_params=dict(temperature=0, max_new_tokens=8, ignore_eos=True),
                return_logprob=True, logprob_start_len=0,
            )
            if accuracy_cases is not None:
                generation_args["input_ids"] = [
                    list(case["input_ids"]) for case in accuracy_cases[wave]]
                generation_args["top_logprobs_num"] = 20
            elif args.family == "checkpoint":
                generation_args["prompt"] = (
                    ["The capital of France is", "2 + 2 ="] if input_wave == 0 else
                    ["The capital of China is", "3 + 3 ="]
                )
            else:
                generation_args["input_ids"] = [
                    [1, 7 + input_wave, 12, 9], [1, 18, 17 + input_wave, 6, 5],
                ]
            result = engine.generate(**generation_args)
            validate_generation(result, vocab_size)
            generations.append(result)
        replay = validate_request_replay(generations) if accuracy_cases is None else None
        cache_probe = {"applicable": False, "reason": "recurrent cache or accuracy profile"}
        cache_family = args.family in ("dense", "moe", "mla") or (
            args.family == "checkpoint" and text_configuration(config).get("model_type") in
            ("qwen3", "qwen2", "llama", "deepseek_v2", "deepseek_v3"))
        if accuracy_cases is None and cache_family:
            cache_probe = probe_cache_reuse(engine, generation_args, vocab_size)
        info = engine.get_server_info()
        validate_mode(info, args.mode)
        validate_lm_head_option(info, fp32_head)
        validate_execution_dtype(info, dtype)
        validate_quantization_option(info, quantization)
        validate_tp_option(info, tp_size)
        validate_memory_option(info, mem_fraction)
    if checkpoint is not None and not audit_checkpoint(args.model, checkpoint)["passed"]:
        raise ValueError("checkpoint changed during model execution")
    write_json(args.out, {
        "artifact": "l15-model-worker", "passed": True, "family": args.family,
        "mode": args.mode, "model": str(args.model),
        "host": getattr(args, "host", "adapter"),
        "weight_evidence": "real-checkpoint" if args.family == "checkpoint" else "dummy",
        "checkpoint_files_sha256": checkpoint,
        "generation_profile": profile, "accuracy_cases": accuracy_cases,
        "enable_fp32_lm_head": fp32_head,
        "dtype": dtype,
        "quantization": quantization,
        "tp_size": tp_size,
        "mem_fraction_static": mem_fraction,
        "model_type": config.get("model_type"),
        "text_model_type": text_configuration(config).get("model_type"),
        "generations": generations, "request_replay": replay,
        "cache_probe": cache_probe, "server_info": info,
    })


def run(args):
    root = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    work = output.parent / (output.stem + "-jobs")
    work.mkdir(parents=True, exist_ok=True)
    families = args.families.split(",")
    modes = args.modes.split(",")
    profile = getattr(args, "profile", "smoke")
    fp32_head = getattr(args, "enable_fp32_lm_head", False)
    dtype = getattr(args, "dtype", "bfloat16")
    quantization = getattr(args, "quantization", None)
    tp_size = getattr(args, "tp_size", 1)
    validate_tp_size(tp_size)
    mem_fraction = getattr(args, "mem_fraction_static", .25)
    validate_memory_fraction(mem_fraction)
    if dtype not in ("bfloat16", "float32"):
        raise ValueError("select bfloat16 or float32 execution")
    if profile == "accuracy" and families != ["checkpoint"]:
        raise ValueError("accuracy profile requires only the real checkpoint family")
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
    host = getattr(args, "host", "adapter")
    runtime = preflight(args.up_tree.resolve(), root, args.binding_manifest.resolve(),
                        host=host, device_count=tp_size)
    checkpoint_inputs = fingerprint(args.checkpoint.resolve()) if "checkpoint" in families else None
    jobs = []
    for family in families:
        if family == "checkpoint":
            model = args.checkpoint.resolve()
            checkpoint = checkpoint_inputs
            runtime.setdefault("checkpoints", {})[str(model)] = checkpoint
            manifest = work / "checkpoint-files.json"
            write_json(manifest, checkpoint)
        else:
            model = work / ("tiny-" + family)
            write_json(model / "config.json", tiny_config(family))
            checkpoint = None
        for mode in modes:
            report = work / f"{family}-{mode}.json"
            report.unlink(missing_ok=True)
            job = run_job(
                [sys.executable, "-m", "tools.l15_model_smoke", "--worker",
                 "--model", str(model), "--family", family, "--mode", mode,
                 "--out", str(report), "--profile", profile, "--dtype", dtype,
                 "--host", host, "--tp-size", str(tp_size),
                 "--mem-fraction-static", str(mem_fraction)]
                + (["--checkpoint-manifest", str(manifest)] if checkpoint is not None else [])
                + (["--enable-fp32-lm-head"] if fp32_head else [])
                + (["--quantization", quantization] if quantization is not None else []),
                cwd=root, env=test_env(args.up_tree.resolve(), root, host=host),
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
                and evidence.get("model") == str(model)
                and evidence.get("host", "adapter") == host
                and evidence.get("weight_evidence") == (
                    "real-checkpoint" if checkpoint is not None else "dummy")
                and evidence.get("generation_profile", "smoke") == profile
                and evidence.get("enable_fp32_lm_head", False) is fp32_head
                and evidence.get("dtype", "bfloat16") == dtype
                and evidence.get("quantization") == quantization
                and type(evidence.get("tp_size", 1)) is int
                and evidence.get("tp_size", 1) == tp_size
                and type(evidence.get("mem_fraction_static", .25)) in (int, float)
                and evidence.get("mem_fraction_static", .25) == mem_fraction
            )
            if checkpoint is not None:
                binding = audit_checkpoint(model, checkpoint)
                job["checkpoint_integrity_after"] = binding
                job["passed"] = (job["passed"] and binding["passed"]
                                 and evidence.get("checkpoint_files_sha256") == checkpoint)
            integrity = audit_source(args.up_tree.resolve(), runtime["upstream"]["commit"])
            job["source_integrity_after"] = integrity
            job["passed"] = job["passed"] and integrity["passed"]
            jobs.append(job)
            write_json(output, {
                "artifact": "l15-model-smoke", "runtime": runtime,
                "passed": len(jobs) == len(families) * len(modes) and all(j["passed"] for j in jobs),
                "complete": len(jobs) == len(families) * len(modes),
                "families": families, "modes": modes, "jobs": jobs,
                "generation_profile": profile,
                "enable_fp32_lm_head": fp32_head,
                "dtype": dtype,
                "quantization": quantization,
                "tp_size": tp_size,
                "mem_fraction_static": mem_fraction,
            })
            print(f"model smoke: {family}/{mode} {'PASS' if job['passed'] else 'FAIL'}", flush=True)
            if not integrity["passed"] or (
                checkpoint is not None and not job["checkpoint_integrity_after"]["passed"]
            ):
                return False  # preserve failure and stop using changed inputs
    return all(job["passed"] for job in jobs)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--up-tree", type=Path)
    parser.add_argument("--families", default="dense,moe,mla,mamba,hybrid,checkpoint")
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--profile", choices=("smoke", "accuracy"), default="smoke")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                        help="official model compute dtype; float32 is an opt-in diagnostic")
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=1,
                        help="tensor parallel size; expose exactly this many idle HCU devices")
    parser.add_argument("--mem-fraction-static", type=float, default=.25,
                        help="official engine memory budget fraction; default remains 0.25")
    parser.add_argument("--quantization",
                        help="official quantization override; default is checkpoint auto-detection")
    parser.add_argument("--enable-fp32-lm-head", action="store_true",
                        help="use the official FP32 output-layer option for precision attribution")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--binding-manifest", type=Path,
                        default=DEFAULT_BINDING_MANIFEST)
    parser.add_argument("--host", choices=("adapter", "adapter+plugin"),
                        default="adapter",
                        help="load only the adapter, or also the installed "
                             "hcu-sglang-plugin dist (SGLANG_PLUGINS=hcu)")
    parser.add_argument("--out", type=Path, default=Path("results/l15-readiness/models.json"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--family", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("eager", "graph"), help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-manifest", type=Path, help=argparse.SUPPRESS)
    return parser


def main():
    parser = build_parser()
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
