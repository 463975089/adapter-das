"""Compare SGLang probabilities only at identical token histories.

Accept records containing ``input_ids`` and a raw ``result`` from /generate.
This CPU-only diagnostic works for eager/graph responses and for incremental
decode versus a fresh prefill of a frozen prefix. It does not establish model
accuracy or verify checkpoint, source, execution mode, or seed provenance.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def _ids(value):
    if (not isinstance(value, list) or not value
            or any(type(token) is not int or token < 0 for token in value)):
        raise ValueError("token IDs must be a nonempty list of nonnegative integers")
    return value


def _probability(row):
    if (not isinstance(row, (list, tuple)) or len(row) < 2
            or type(row[0]) not in (int, float) or not math.isfinite(row[0])
            or row[0] > 0 or type(row[1]) is not int or row[1] < 0):
        raise ValueError("requires finite nonpositive logprobs and integer token IDs")
    return row[1], float(row[0])


def _record(record):
    if not isinstance(record, dict) or not isinstance(record.get("result"), dict):
        raise ValueError("requires an input_ids/result record")
    prompt = _ids(record.get("input_ids"))
    result = record["result"]
    output = _ids(result.get("output_ids"))
    meta = result.get("meta_info")
    if not isinstance(meta, dict):
        raise ValueError("missing response metadata")
    for key, expected in (("prompt_tokens", len(prompt)),
                          ("completion_tokens", len(output))):
        if type(meta.get(key)) is not int or meta[key] != expected:
            raise ValueError("response token counts do not match IDs")
    cached = meta.get("cached_tokens")
    if type(cached) is not int or not 0 <= cached <= len(prompt):
        raise ValueError("missing or invalid cached token count")
    selected, top = meta.get("output_token_logprobs"), meta.get("output_top_logprobs")
    if (not isinstance(selected, list) or not isinstance(top, list)
            or len(selected) != len(output) or len(top) != len(output)):
        raise ValueError("requires selected and top logprobs at every output position")
    rows = []
    for position, token in enumerate(output):
        selected_id, selected_score = _probability(selected[position])
        if selected_id != token:
            raise ValueError("selected logprob token does not match output ID")
        head = top[position]
        if not isinstance(head, list) or not head:
            raise ValueError("each output position requires a nonempty top-logprob list")
        pairs = [_probability(row) for row in head]
        scores = dict(pairs)
        if len(scores) != len(pairs):
            raise ValueError("duplicate top-logprob token")
        if token in scores and scores[token] != selected_score:
            raise ValueError("selected and top logprobs disagree for the same token")
        top_ids = set(scores)
        scores[token] = selected_score
        rows.append((tuple(prompt + output[:position]), {
            "position": position, "token": token, "selected": selected_score,
            "top_ids": top_ids, "scores": scores,
        }))
    return rows, cached


def compare(left, right):
    """Align complete histories, never comparing different generated prefixes."""
    left_rows, left_cached = _record(left)
    right_rows, right_cached = _record(right)
    right_index = dict(right_rows)
    aligned = []
    for history, a in left_rows:
        b = right_index.get(history)
        if b is None:
            continue
        common = sorted(a["scores"].keys() & b["scores"].keys())
        deltas = [
            {"token_id": token, "left_logprob": a["scores"][token],
             "right_logprob": b["scores"][token],
             "absolute_delta": abs(a["scores"][token] - b["scores"][token])}
            for token in common
        ]
        aligned.append({
            "history_tokens": len(history),
            "left_output_position": a["position"],
            "right_output_position": b["position"],
            "left_selected_token": a["token"], "right_selected_token": b["token"],
            "selected_ids_equal": a["token"] == b["token"],
            "selected_logprob_delta": (
                abs(a["selected"] - b["selected"]) if a["token"] == b["token"] else None),
            "left_top_count": len(a["top_ids"]), "right_top_count": len(b["top_ids"]),
            "shared_top_count": len(a["top_ids"] & b["top_ids"]),
            "shared_reported_count": len(common),
            "max_shared_logprob_delta": (
                max(row["absolute_delta"] for row in deltas) if deltas else None),
            "shared_logprobs": deltas,
        })
    if not aligned:
        raise ValueError("responses have no identical prediction history")
    first_difference = next((
        {"left_output_position": row["left_output_position"],
         "right_output_position": row["right_output_position"]}
        for row in aligned
        if not row["selected_ids_equal"]
        or (row["max_shared_logprob_delta"] or 0) > 0
    ), None)
    return {
        "artifact": "l15-decode-consistency-diagnostic",
        "schema_version": 1,
        "model_acceptance": False,
        "scope": (
            "Reported probabilities at identical histories only; no independent "
            "reference, full-vocabulary comparison, or provenance verification."),
        "left_cached_tokens": left_cached, "right_cached_tokens": right_cached,
        "left_output_count": len(left_rows), "right_output_count": len(right_rows),
        "aligned_count": len(aligned),
        "left_unaligned_count": len(left_rows) - len(aligned),
        "right_unaligned_count": len(right_rows) - len(aligned),
        "first_observed_difference": first_difference,
        "positions": aligned,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = [path.read_bytes() for path in (args.left, args.right)]
    report = compare(*(json.loads(value) for value in raw))
    report["input_sha256"] = {
        side: hashlib.sha256(value).hexdigest()
        for side, value in zip(("left", "right"), raw)
    }
    # Preserve existing evidence, including when --out names either input.
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
