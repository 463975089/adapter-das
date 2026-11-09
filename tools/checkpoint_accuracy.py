# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fixed real-checkpoint cases and BF16 probability comparison policy."""

POLICY = {
    "max_logprob_error": .15,
    "max_greedy_regret": .15,
    "max_top20_logprob_error": .15,
    "max_top20_tail_jsd": .002,
    "max_top20_tail_mass_error": .03,
    "max_top20_missed_mass": .02,
}
POLICY_NAME = "bf16-cpu-fp32-top20-v1"
# A large-vocabulary FP32 log_softmax can accumulate a small mass drift when
# its probabilities are re-summed in float64. The top20/tail vectors below
# are normalized before JSD and mass comparisons; reject only larger drift.
REFERENCE_MASS_TOLERANCE = 1e-3


def cases(tokenizer):
    """Short bilingual inputs, exact 512/2048 lengths, reuse and changed code."""
    def short(label, prompt):
        if tokenizer.chat_template:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False,
            )
        else:
            ids = tokenizer.encode(prompt + "\nAnswer:", add_special_tokens=False)
        return {"label": label, "input_ids": ids}

    def document(length, code, label):
        prefix = tokenizer.encode(
            f"Document: The access code is {code}.\n", add_special_tokens=False)
        suffix = tokenizer.encode(
            "\nQuestion: What is the access code?\nAnswer:", add_special_tokens=False)
        fill = tokenizer.encode(
            "The document contains routine notes about schedules and supplies.\n",
            add_special_tokens=False,
        )
        remaining = length - len(prefix) - len(suffix)
        if not fill or remaining < 0:
            raise ValueError("tokenizer cannot construct fixed-length accuracy case")
        ids = prefix + (fill * ((remaining + len(fill) - 1) // len(fill)))[:remaining] + suffix
        return {"label": label, "input_ids": ids}

    return [
        [short("short-en", "Answer with one word. What is the capital of France?"),
         short("short-zh", "请用一个词回答：中国的首都是哪里？")],
        [document(512, "7319", "context-512"),
         document(2048, "6248", "context-2048")],
        [document(512, "7319", "reuse-512"),
         document(2048, "9137", "changed-2048")],
    ]


def compare_top20(reference, observed):
    """Compare the same 20 token categories plus all remaining mass.

    This is a coarsened distribution, not a full-vocabulary divergence.
    Ranks within near ties are free; token identities and probabilities matter.
    """
    import torch

    if len(observed) != len(reference) or any(len(row) != 20 for row in observed):
        raise ValueError("requires 20 probabilities at every generated position")
    try:
        ids = torch.tensor([[v[1] for v in row] for row in observed], dtype=torch.long)
        scores = torch.tensor([[v[0] for v in row] for row in observed], dtype=torch.float64)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("invalid top20 probability evidence") from exc
    if (reference.ndim != 2 or not torch.isfinite(reference).all()
            or (reference > 0).any()
            or (reference.double().exp().sum(-1) - 1).abs().max()
            > REFERENCE_MASS_TOLERANCE
            or not torch.isfinite(scores).all() or (scores > 0).any()
            or ids.min() < 0 or ids.max() >= reference.shape[1]
            or any(len({v[1] for v in row}) != 20 for row in observed)
            or any(type(v[1]) is not int for row in observed for v in row)):
        raise ValueError("invalid top20 token identities or logprobs")
    selected = reference.double().gather(1, ids)
    head, ref_head = scores.exp(), selected.exp()
    if (head.sum(-1) > 1 + 1e-5).any():
        raise ValueError("top20 probability mass exceeds one")
    p = torch.cat((head, (1 - head.sum(-1, keepdim=True)).clamp_min(0)), -1)
    q = torch.cat((ref_head, (1 - ref_head.sum(-1, keepdim=True)).clamp_min(0)), -1)
    p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
    midpoint = (p + q) / 2
    jsd = .5 * (torch.special.xlogy(p, p / midpoint).nan_to_num(0).sum(-1)
                + torch.special.xlogy(q, q / midpoint).nan_to_num(0).sum(-1))
    return {
        "max_top20_logprob_error": (scores - selected).abs().max().item(),
        "max_top20_tail_jsd": jsd.max().clamp_min(0).item(),
        "max_top20_tail_mass_error": (p[:, -1] - q[:, -1]).abs().max().item(),
        "max_top20_missed_mass": (
            reference.double().topk(20, dim=-1).values.exp().sum(-1) - ref_head.sum(-1)
        ).clamp_min(0).max().item(),
    }


def validate_cases(batches):
    labels = (("short-en", "short-zh"), ("context-512", "context-2048"),
              ("reuse-512", "changed-2048"))
    if not isinstance(batches, list) or len(batches) != 3:
        raise ValueError("accuracy requires all three case batches")
    for wave, batch in enumerate(batches):
        if len(batch) != 2:
            raise ValueError("accuracy requires two requests per batch")
        for index, case in enumerate(batch):
            ids = case.get("input_ids", [])
            if (case.get("label") != labels[wave][index] or not ids
                    or any(type(token) is not int or token < 0 for token in ids)
                    or (wave == 0 and len(ids) >= 512)
                    or (wave > 0 and len(ids) != (512 if index == 0 else 2048))):
                raise ValueError("accuracy case label, length or token IDs changed")
    if (batches[1][0]["input_ids"] != batches[2][0]["input_ids"]
            or batches[1][1]["input_ids"] == batches[2][1]["input_ids"]):
        raise ValueError("accuracy requires both reused and changed input")


def verdict(rows):
    import math

    if not rows:
        raise ValueError("accuracy evidence cannot be empty")
    if any(not math.isfinite(row[key]) or row[key] < 0 for row in rows for key in POLICY):
        raise ValueError("accuracy metrics must be finite and nonnegative")
    maxima = {key: max(row[key] for row in rows) for key in POLICY}
    return {"policy": POLICY_NAME, "limits": dict(POLICY), "maxima": maxima,
            "passed": all(maxima[key] <= bound for key, bound in POLICY.items())}
