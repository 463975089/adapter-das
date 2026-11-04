# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU FP32/BF16 attribution on frozen, byte-bound HCU accuracy requests.

This diagnostic does not change or replace the FP32 accuracy gate. Run in a
fresh HOSTnone process with all GPUs hidden.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys

from tools.checkpoint_accuracy import compare_top20
from tools.checkpoint_files import fingerprint
from tools.checkpoint_model import reference_loader, text_configuration, vocabulary_size
from tools.l15_checkpoint_reference import execution_binding, requests
from tools.l15_runtime import write_json


def compare_dtypes(fp32, bf16, observed):
    """Use identical prefixes, with independent BF16 heads and HCU label anchors."""
    import torch

    if fp32.shape != bf16.shape:
        raise ValueError("dtype reference shapes differ")
    hcu_fp32 = compare_top20(fp32, observed)
    hcu_bf16 = compare_top20(bf16, observed)
    values, tokens = bf16.topk(20, dim=-1)
    bf16_head = [
        [[score, token, None] for score, token in zip(scores, ids)]
        for scores, ids in zip(values.tolist(), tokens.tolist())
    ]
    cpu = compare_top20(fp32, bf16_head)
    ids = torch.tensor([[v[1] for v in row] for row in observed], dtype=torch.long)
    hcu = torch.tensor([[v[0] for v in row] for row in observed], dtype=torch.float64)
    single, reduced = fp32.double().gather(1, ids), bf16.double().gather(1, ids)

    def outlier(error):
        index = int(error.argmax())
        position, rank = divmod(index, 20)
        return {
            "output_position": position, "observed_rank": rank + 1,
            "token_id": int(ids[position, rank]),
            "hcu_logprob": float(hcu[position, rank]),
            "cpu_fp32_logprob": float(single[position, rank]),
            "cpu_bf16_logprob": float(reduced[position, rank]),
            "absolute_error": float(error[position, rank]),
        }

    return {
        "hcu_vs_cpu_fp32": hcu_fp32,
        "hcu_vs_cpu_bf16": hcu_bf16,
        "cpu_bf16_vs_cpu_fp32": cpu,
        "worst_hcu_vs_fp32": outlier((hcu - single).abs()),
        "worst_cpu_dtype_on_hcu_labels": outlier((reduced - single).abs()),
    }


def probabilities(model, row, vocab_size):
    import torch

    prompt, output = row["prompt_ids"], row["output_ids"]
    ids = torch.tensor([prompt + output], dtype=torch.long)
    if ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError("token outside reference vocabulary")
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        logits = model(input_ids=ids, use_cache=False,
                       logits_to_keep=len(output) + 1).logits[0, :-1]
    else:
        logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt) - 1:-1]
    if len(logits) != len(output):
        raise ValueError("reference did not return every generated position")
    return logits.float().log_softmax(-1)


def run(args):
    if args.out.resolve() == args.smoke_report.resolve():
        raise ValueError("diagnostic output must preserve original smoke evidence")
    from hcu_adapter import hcu_env

    if hcu_env.resolve("HCU_ADAPTER_HOST")[0] != "none":
        raise ValueError("use a fresh HOSTnone process with GPUs hidden")
    import torch
    import transformers

    if torch.cuda.is_available() or any(
        name == "sglang" or name.startswith(("sglang.", "hcu_adapter.bridges."))
        for name in sys.modules
    ):
        raise ValueError("CPU diagnostic must not expose GPUs or host bridges")
    torch.set_num_threads(args.threads)
    raw = args.smoke_report.read_bytes()
    document = json.loads(raw)
    model_path, rows = requests(document)
    bound = execution_binding(document, model_path)
    if document.get("generation_profile") != "accuracy" or bound is None:
        raise ValueError("dtype attribution requires byte-bound accuracy requests")
    if model_path == args.out.resolve() or model_path in args.out.resolve().parents:
        raise ValueError("diagnostic output must be outside checkpoint directory")
    config = json.loads((model_path / "config.json").read_text())
    if (document.get("quantization") is not None or config.get("quantization_config")
            or text_configuration(config).get("quantization_config")):
        raise ValueError("dtype attribution does not support quantized checkpoints or execution")
    before = fingerprint(model_path)
    if before != bound:
        raise ValueError("reference checkpoint bytes differ from HCU execution")
    vocab_size = vocabulary_size(config)
    cached = {}
    buffers = {}
    with torch.inference_mode():
        for dtype in (torch.float32, torch.bfloat16):
            # Whole-model .to(bfloat16) also rounds FP32 RoPE buffers.
            # Let the official loader initialize each dtype independently.
            model = reference_loader(config, transformers).from_pretrained(
                str(model_path), dtype=dtype, attn_implementation="eager",
                local_files_only=True,
            ).eval()
            model_class = type(model).__name__
            if any(p.device.type != "cpu" or p.dtype != dtype for p in model.parameters()):
                raise ValueError("reference weights have unexpected device or dtype")
            buffers[str(dtype)] = {
                name: {
                    "dtype": str(value.dtype), "shape": list(value.shape),
                    "sha256": hashlib.sha256(
                        value.detach().cpu().contiguous().reshape(-1)
                        .view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
                for name, value in model.named_buffers() if value.is_floating_point()
            }
            # Deduplicate eager/graph and reused requests only when every
            # prompt/output token agrees. Each dtype executes independently.
            for row in rows:
                key = tuple(row["prompt_ids"] + row["output_ids"])
                if key not in cached:
                    cached[key] = {}
                if str(dtype) not in cached[key]:
                    cached[key][str(dtype)] = probabilities(model, row, vocab_size)
                    print(f"{dtype}: {row['mode']}/{row['case']}, "
                          f"{len(row['prompt_ids'])} prompt tokens", flush=True)
            del model
            gc.collect()
        compared = []
        for row in rows:
            pair = cached[tuple(row["prompt_ids"] + row["output_ids"])]
            compared.append({
                **row,
                **compare_dtypes(pair["torch.float32"], pair["torch.bfloat16"],
                                 row["observed_top_logprobs"]),
            })
    if fingerprint(model_path) != before or args.smoke_report.read_bytes() != raw:
        raise ValueError("checkpoint or smoke evidence changed during dtype attribution")
    write_json(args.out, {
        "artifact": "l15-checkpoint-dtype-diagnostic", "diagnostic_only": True,
        "smoke_report": str(args.smoke_report.resolve()),
        "smoke_sha256": hashlib.sha256(raw).hexdigest(),
        "model": str(model_path), "checkpoint_binding": "matched-hcu-execution",
        "reference_files_sha256": before,
        "enable_fp32_lm_head": document.get("enable_fp32_lm_head", False),
        "execution_dtype": document.get("dtype", "bfloat16"),
        "execution_quantization": document.get("quantization"),
        "reference": {
            "device": "cpu", "attention": "eager",
            "dtypes": ["float32", "bfloat16"], "torch": torch.__version__,
            "transformers": transformers.__version__, "model_class": model_class,
            "text_model_type": text_configuration(config).get("model_type"),
            "threads": args.threads, "bf16_lm_head": "bfloat16",
            "loading": "independent-from-pretrained-per-dtype",
            "floating_buffers": buffers,
        },
        "limitations": [
            "Both CPU references teacher-force the same HCU-generated sequences; "
            "they do not perform independent greedy generation.",
            "CPU BF16 includes the output layer and uses Transformers eager; "
            "it is not an implementation-equivalent SGLang or HCU baseline.",
            "Top20+tail is a coarsened distribution. Dtype differences neither "
            "prove an HCU bug nor excuse a failed FP32 accuracy gate.",
            "Output positions are zero-based; observed ranks are one-based.",
        ],
        "rows": compared,
    })
    for comparison in ("hcu_vs_cpu_fp32", "hcu_vs_cpu_bf16", "cpu_bf16_vs_cpu_fp32"):
        print(comparison, max(row[comparison]["max_top20_logprob_error"] for row in compared),
              flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return 0 if run(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
