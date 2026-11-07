"""Validate the actual EP4 recipe with community ServerArgs, without weights."""

import argparse
import os
from pathlib import Path
import re
import shlex
from string import Template


def check(recipe, model_path):
    source = recipe.read_text()
    command = source.split("exec python -m sglang.launch_server", 1)[1]
    command = Template(command.replace("\\\n", " ")).substitute(
        MODEL_PATH=model_path, HOST="127.0.0.1", PORT="30000"
    )
    capacity = int(re.search(
        r"^export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=(\d+)$",
        source, re.MULTILINE,
    ).group(1))
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.server_args import prepare_server_args

    args = prepare_server_args(shlex.split(command))
    # Match Engine._launch_subprocesses: parsing alone leaves late defaults
    # such as served_model_name and model-specific page size unresolved.
    args.resolve_once()
    args.check_server_args()
    cfg = resolving_view(args)
    assert cfg.chunked_prefill_size <= capacity, "prefill exceeds DeepEP LL capacity"
    assert cfg.max_running_requests <= capacity, "decode exceeds DeepEP LL capacity"
    assert capacity <= 1024, "capacity exceeds community DeepEP LL limit"
    print(
        f"RECIPE_PREFLIGHT_PASS page_size={cfg.page_size} "
        f"chunk={cfg.chunked_prefill_size} dispatch_capacity={capacity}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe", type=Path,
        default=Path(__file__).with_name("_launch_dsv41_int4_ep4_pristine.sh"),
    )
    parser.add_argument(
        "--model-path", default=os.environ.get(
            "MODEL_PATH", "/model/DeepSeek-V4.1-Flash-Channel-INT4-w4a8"
        ),
    )
    options = parser.parse_args()
    check(options.recipe, options.model_path)
