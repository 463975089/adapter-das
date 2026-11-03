"""Exploratory CPU FP32 reference for a successful real-checkpoint smoke.

Run with HCU_ADAPTER_HOST=none and all GPUs hidden. This reports sampled-token
logprob differences and greedy regret; it is not a model-accuracy acceptance
gate. In particular, historical smoke reports do not fingerprint weight bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from tools.l15_model_smoke import validate_generation, validate_mode
from tools.l15_runtime import write_json


def requests(document):
    if (document.get("artifact") != "l15-model-smoke"
            or document.get("passed") is not True
            or document.get("complete") is not True
            or document.get("families") != ["checkpoint"]
            or len(document.get("modes", [])) != 2
            or set(document.get("modes", [])) != {"eager", "graph"}
            or len(document.get("jobs", [])) != 2
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
                or mode not in {"eager", "graph"} or mode in modes):
            raise ValueError("invalid or duplicate checkpoint worker")
        modes.add(mode)
        models.add(str(Path(evidence["model"]).resolve()))
        validate_mode(evidence.get("server_info", {}), mode)
        waves = evidence.get("generations", [])
        if len(waves) != 2:
            raise ValueError("requires both input waves")
        for wave, batch in enumerate(waves):
            # Vocabulary bounds are checked again against the loaded reference.
            validate_generation(batch, 2**63)
            for request, item in enumerate(batch):
                info = item["meta_info"]
                prompt = [entry[1] for entry in info.get("input_token_logprobs", [])]
                if (not prompt or len(prompt) != info.get("prompt_tokens")
                        or info.get("cached_tokens") != 0
                        or any(type(token) is not int or token < 0 for token in prompt)):
                    raise ValueError("requires complete uncached prompt token IDs")
                rows.append({
                    "mode": mode, "wave": wave, "request": request,
                    "prompt_ids": prompt, "output_ids": item["output_ids"],
                    "observed_logprobs": [v[0] for v in info["output_token_logprobs"]],
                })
    if len(models) != 1:
        raise ValueError("workers must use the same checkpoint")
    return Path(models.pop()), rows


def fingerprint(model):
    files = sorted({*model.glob("*.safetensors"), *model.glob("*.json")})
    if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
        raise ValueError("reference requires local config and real safetensors")
    result = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[path.name] = digest.hexdigest()
    return result


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
    from transformers import AutoModelForCausalLM

    if torch.cuda.is_available() or any(
        name == "sglang" or name.startswith(("sglang.", "hcu_adapter.bridges."))
        for name in sys.modules
    ):
        raise ValueError("CPU reference must not expose a GPU or install host bridges")
    torch.set_num_threads(args.threads)
    raw = args.smoke_report.read_bytes()
    model_path, rows = requests(json.loads(raw))
    config = json.loads((model_path / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError("this FP32 probe does not validate quantized checkpoints")
    before = fingerprint(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.float32, attn_implementation="eager",
        local_files_only=True,
    ).eval()
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("reference weights must be CPU FP32")
    with torch.inference_mode():
        for row in rows:
            prompt, output = row["prompt_ids"], row["output_ids"]
            ids = torch.tensor([prompt + output], dtype=torch.long)
            if ids.min() < 0 or ids.max() >= model.config.vocab_size:
                raise ValueError("token outside reference vocabulary")
            logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt) - 1:-1]
            row.update(compare(logits.float().log_softmax(-1), output,
                               row["observed_logprobs"]))
    if fingerprint(model_path) != before or args.smoke_report.read_bytes() != raw:
        raise ValueError("checkpoint or smoke evidence changed during reference execution")
    write_json(args.out, {
        "artifact": "l15-checkpoint-reference-probe",
        "acceptance": "exploratory-short-prompts",
        "smoke_report": str(args.smoke_report.resolve()),
        "smoke_sha256": hashlib.sha256(raw).hexdigest(),
        "model": str(model_path), "reference_files_sha256": before,
        "reference": {"torch": torch.__version__, "transformers": transformers.__version__,
                      "device": "cpu", "dtype": "float32", "attention": "eager"},
        "limitations": [
            "Weight fingerprints were collected at reference time; historical smoke "
            "reports do not bind the HCU execution to those weight bytes.",
            "Only sampled-token probabilities on short prompts are compared; "
            "this is not full-distribution, long-context or model-accuracy acceptance.",
        ],
        "rows": rows,
    })
    print(f"CPU reference: {sum(len(r['output_ids']) for r in rows)} positions, "
          f"max logprob error={max(r['max_logprob_error'] for r in rows):.6f}, "
          f"max greedy regret={max(r['max_greedy_regret'] for r in rows):.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
