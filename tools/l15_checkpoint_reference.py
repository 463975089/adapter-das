# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU FP32 reference for real-checkpoint generation evidence.

Run with HCU_ADAPTER_HOST=none and all GPUs hidden. This reports sampled-token
logprob differences and greedy regret. Smoke inputs remain exploratory;
the bound accuracy profile also gates fixed cases on top20+tail probabilities.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

from tools.l15_model_smoke import (
    validate_execution_dtype, validate_generation, validate_lm_head_option, validate_mode,
    validate_quantization_option,
)
from tools.l15_runtime import write_json
from tools.checkpoint_files import fingerprint
from tools.checkpoint_model import reference_loader, text_configuration, vocabulary_size


def requests(document):
    profile = document.get("generation_profile", "smoke")
    fp32_head = document.get("enable_fp32_lm_head", False)
    dtype = document.get("dtype", "bfloat16")
    quantization = document.get("quantization")
    if (document.get("artifact") != "l15-model-smoke"
            or document.get("passed") is not True
            or document.get("complete") is not True
            or document.get("families") != ["checkpoint"]
            or len(document.get("modes", [])) != 2
            or set(document.get("modes", [])) != {"eager", "graph"}
            or len(document.get("jobs", [])) != 2
            or profile not in {"smoke", "accuracy"}
            or type(fp32_head) is not bool
            or dtype not in ("bfloat16", "float32")
            or document.get("runtime", {}).get("source_integrity", {}).get("passed") is not True):
        raise ValueError("requires successful pristine-source eager/graph checkpoint evidence")
    rows, models, modes = [], set(), set()
    for job in document["jobs"]:
        evidence = job.get("evidence", {})
        mode = job.get("mode")
        if (job.get("passed") is not True or job.get("returncode") != 0
                or job.get("timed_out") is not False
                or evidence.get("artifact") != "l15-model-worker"
                or evidence.get("passed") is not True
                or evidence.get("weight_evidence") != "real-checkpoint"
                or evidence.get("family") != "checkpoint"
                or evidence.get("mode") != mode
                or evidence.get("generation_profile", "smoke") != profile
                or evidence.get("enable_fp32_lm_head", False) is not fp32_head
                or evidence.get("dtype", "bfloat16") != dtype
                or evidence.get("quantization") != quantization
                or mode not in {"eager", "graph"} or mode in modes):
            raise ValueError("invalid or duplicate checkpoint worker")
        modes.add(mode)
        models.add(str(Path(evidence["model"]).resolve()))
        validate_mode(evidence.get("server_info", {}), mode)
        validate_lm_head_option(evidence.get("server_info", {}), fp32_head)
        validate_execution_dtype(evidence.get("server_info", {}), dtype)
        validate_quantization_option(evidence.get("server_info", {}), quantization)
        waves = evidence.get("generations", [])
        cases = evidence.get("accuracy_cases")
        if profile == "accuracy":
            from tools.checkpoint_accuracy import validate_cases

            validate_cases(cases)
            if job.get("source_integrity_after", {}).get("passed") is not True:
                raise ValueError("accuracy requires source integrity after every worker")
        if len(waves) != (3 if profile == "accuracy" else 2):
            raise ValueError("requires both input waves")
        for wave, batch in enumerate(waves):
            # Vocabulary bounds are checked again against the loaded reference.
            validate_generation(batch, 2**63)
            for request, item in enumerate(batch):
                info = item["meta_info"]
                prompt = (cases[wave][request]["input_ids"] if profile == "accuracy" else
                          [entry[1] for entry in info.get("input_token_logprobs", [])])
                if (not prompt or len(prompt) != info.get("prompt_tokens")
                        or type(info.get("cached_tokens")) is not int
                        or not 0 <= info["cached_tokens"] <= len(prompt)
                        or (profile == "smoke" and info["cached_tokens"] != 0)
                        or any(type(token) is not int or token < 0 for token in prompt)):
                    raise ValueError("requires complete uncached prompt token IDs")
                rows.append({
                    "mode": mode, "wave": wave, "request": request,
                    "prompt_ids": prompt, "output_ids": item["output_ids"],
                    "observed_logprobs": [v[0] for v in info["output_token_logprobs"]],
                })
                if profile == "accuracy":
                    top = info.get("output_top_logprobs", [])
                    if len(top) != len(item["output_ids"]):
                        raise ValueError("missing generated top20 probabilities")
                    for token, logprob, head in zip(
                        item["output_ids"], rows[-1]["observed_logprobs"], top
                    ):
                        values = {v[1]: v[0] for v in head}
                        if token not in values or abs(values[token] - logprob) > 1e-5:
                            raise ValueError("sampled token and top20 probabilities disagree")
                    rows[-1].update(case=cases[wave][request]["label"],
                                    observed_top_logprobs=top)
    if len(models) != 1:
        raise ValueError("workers must use the same checkpoint")
    return Path(models.pop()), rows


def execution_binding(document, model):
    expected = document.get("runtime", {}).get("checkpoints", {}).get(str(model))
    if expected is None:
        # Historical reports remain explicitly exploratory and unbound.
        return None
    if not isinstance(expected, dict) or not expected:
        raise ValueError("invalid HCU checkpoint manifest")
    for job in document["jobs"]:
        checked = job.get("checkpoint_integrity_after", {})
        if (job["evidence"].get("checkpoint_files_sha256") != expected
                or checked.get("passed") is not True
                or checked.get("files_sha256") != expected):
            raise ValueError("inconsistent HCU checkpoint byte binding")
    return expected


def compare(logprobs, output_ids, observed):
    import torch

    tokens = torch.tensor(output_ids, dtype=torch.long)
    actual = torch.tensor(observed, dtype=torch.float32)
    if (logprobs.ndim != 2 or len(tokens) != len(logprobs)
            or len(actual) != len(tokens) or not len(tokens)
            or not torch.isfinite(logprobs).all() or not torch.isfinite(actual).all()
            or tokens.min() < 0 or tokens.max() >= logprobs.shape[-1]):
        raise ValueError("invalid reference distribution or sampled token evidence")
    selected = logprobs.gather(1, tokens[:, None]).flatten()
    error = (actual - selected).abs()
    regret = logprobs.max(-1).values - selected
    return {
        "reference_logprobs": selected.tolist(), "logprob_errors": error.tolist(),
        "greedy_regrets": regret.tolist(), "max_logprob_error": error.max().item(),
        "mean_logprob_error": error.mean().item(), "max_greedy_regret": regret.max().item(),
        "reference_top1_matches": int((logprobs.argmax(-1) == tokens).sum()),
    }


def run(args):
    if args.out.resolve() == args.smoke_report.resolve():
        raise ValueError("reference output must preserve the original smoke report")
    from hcu_adapter import hcu_env

    if hcu_env.resolve("HCU_ADAPTER_HOST")[0] != "none":
        raise ValueError("start a fresh process with HCU_ADAPTER_HOST=none and GPUs hidden")
    import torch
    import transformers

    if torch.cuda.is_available() or any(
        name == "sglang" or name.startswith(("sglang.", "hcu_adapter.bridges."))
        for name in sys.modules
    ):
        raise ValueError("CPU reference must not expose a GPU or install host bridges")
    torch.set_num_threads(args.threads)
    raw = args.smoke_report.read_bytes()
    document = json.loads(raw)
    model_path, rows = requests(document)
    bound = execution_binding(document, model_path)
    accuracy = document.get("generation_profile") == "accuracy"
    if accuracy and bound is None:
        raise ValueError("accuracy requires checkpoint bytes bound to HCU execution")
    if model_path == args.out.resolve() or model_path in args.out.resolve().parents:
        raise ValueError("reference output must be outside the checkpoint directory")
    config = json.loads((model_path / "config.json").read_text())
    vocab_size = vocabulary_size(config)
    quantized = (document.get("quantization") is not None or config.get("quantization_config")
                 or text_configuration(config).get("quantization_config"))
    reference_kind = getattr(args, "quantized_reference", None)
    if quantized and reference_kind is None:
        raise ValueError("this FP32 probe does not validate quantized checkpoints or execution")
    if reference_kind is not None:
        from tools.checkpoint_quantization_reference import REFERENCE_KIND, validate_configuration

        if reference_kind != REFERENCE_KIND:
            raise ValueError("unsupported quantized reference implementation")
        validate_configuration(config)
        if document.get("quantization") not in (None, "compressed-tensors"):
            raise ValueError("execution quantization does not match the CT FP8 reference")
        if bound is None:
            raise ValueError("quantized reference requires checkpoint bytes bound to HCU execution")
    before = fingerprint(model_path)
    if bound is not None and before != bound:
        raise ValueError("reference checkpoint bytes differ from HCU execution")
    quantization_reference, qdq_audit = None, None
    if reference_kind is not None:
        from tools.checkpoint_quantization_reference import load_reference

        model, quantization_reference, qdq_audit = load_reference(
            model_path, config, transformers, bound,
        )
    else:
        model = reference_loader(config, transformers).from_pretrained(
            str(model_path), dtype=torch.float32, attn_implementation="eager",
            local_files_only=True,
        ).eval()
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("reference weights must be CPU FP32")
    with torch.inference_mode():
        cache = {}
        for row in rows:
            prompt, output = row["prompt_ids"], row["output_ids"]
            ids = torch.tensor([prompt + output], dtype=torch.long)
            if ids.min() < 0 or ids.max() >= vocab_size:
                raise ValueError("token outside reference vocabulary")
            key = tuple(prompt + output)
            if key not in cache:
                # Both validated Dense reference models support this official
                # argument. Keep the general probe path for other architectures.
                import inspect
                with qdq_audit.observe() if qdq_audit is not None else nullcontext():
                    if "logits_to_keep" in inspect.signature(model.forward).parameters:
                        logits = model(input_ids=ids, use_cache=False,
                                       logits_to_keep=len(output) + 1).logits[0, :-1]
                    else:
                        logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt) - 1:-1]
                cache[key] = logits.float().log_softmax(-1)
            probabilities = cache[key]
            row.update(compare(probabilities, output,
                               row["observed_logprobs"]))
            if accuracy:
                from tools.checkpoint_accuracy import compare_top20

                row.update(compare_top20(probabilities, row["observed_top_logprobs"]))
    if fingerprint(model_path) != before or args.smoke_report.read_bytes() != raw:
        raise ValueError("checkpoint or smoke evidence changed during reference execution")
    decision = {}
    if accuracy:
        from tools.checkpoint_accuracy import verdict

        decision = verdict(rows)
    if qdq_audit is not None:
        quantization_reference["forwards"] = qdq_audit.forwards
    write_json(args.out, {
        "artifact": "l15-checkpoint-accuracy" if accuracy else "l15-checkpoint-reference-probe",
        "acceptance": "fixed-input-top20-tail-distribution" if accuracy else "exploratory-short-prompts",
        "smoke_report": str(args.smoke_report.resolve()),
        "smoke_sha256": hashlib.sha256(raw).hexdigest(),
        "model": str(model_path), "reference_files_sha256": before,
        "checkpoint_binding": "matched-hcu-execution" if bound is not None else "legacy-unbound",
        "enable_fp32_lm_head": document.get("enable_fp32_lm_head", False),
        "execution_dtype": document.get("dtype", "bfloat16"),
        "execution_quantization": document.get("quantization"),
        "quantization_reference": quantization_reference,
        "reference": {"torch": torch.__version__, "transformers": transformers.__version__,
                      "device": "cpu", "dtype": "float32", "attention": "eager",
                      "model_class": type(model).__name__,
                      "text_model_type": text_configuration(config).get("model_type")},
        "limitations": ([
            "Historical smoke did not bind weight bytes to HCU execution; "
            "reference-time fingerprints cannot establish that binding.",
        ] if bound is None else []) + ([
            "The fixed bilingual/512/2048/reuse cases compare 20 token labels plus "
            "the remaining mass, not a full-vocabulary divergence or arbitrary model support.",
        ] if accuracy else [
            "Only sampled-token probabilities on short prompts are compared; "
            "this is not full-distribution, long-context or model-accuracy acceptance.",
        ]),
        "rows": rows, **decision,
    })
    print(f"CPU reference: {sum(len(r['output_ids']) for r in rows)} positions, "
          f"max logprob error={max(r['max_logprob_error'] for r in rows):.6f}, "
          f"max greedy regret={max(r['max_greedy_regret'] for r in rows):.6f}")
    if accuracy:
        print(f"fixed-case accuracy: {'PASS' if decision['passed'] else 'FAIL'} "
              f"({decision['policy']})")
    return decision.get("passed", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--quantized-reference", choices=["ct-fp8-token-channel"],
        help="explicit independent CPU QDQ reference for the validated Dense Qwen3 CT FP8 format",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return 0 if run(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
