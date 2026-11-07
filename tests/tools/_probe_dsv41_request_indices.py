"""Real community aliases and spawn bootstrap; no model weights are loaded.

Run with HCU_ADAPTER_HOST=sglang and the pristine community tree on PYTHONPATH.
"""

import multiprocessing as mp
from types import SimpleNamespace


def check():
    import torch
    from sglang.srt.layers.attention.dsv4 import dsv41_sparse
    from sglang.srt.layers.attention import deepseek_v4_backend
    from hcu_adapter import dsv41_request_indices as bridge

    alias = deepseek_v4_backend.token_req_indices
    assert alias is dsv41_sparse.token_req_indices
    assert hasattr(alias, bridge._MARKER), "real bootstrap did not install bridge"
    batch = SimpleNamespace(
        req_pool_indices=torch.tensor([16], device="cuda", dtype=torch.int32),
        extend_seq_lens=torch.tensor([6], device="cuda", dtype=torch.int32),
        forward_mode=SimpleNamespace(
            is_decode=lambda: False,
            is_target_verify=lambda: False,
            is_extend=lambda: True,
        ),
    )
    assert alias(batch, num_tokens=8).cpu().tolist() == [16] * 6 + [0, 0]
    print("REAL_UPSTREAM_ALIAS_PASS", flush=True)


if __name__ == "__main__":
    check()
    child = mp.get_context("spawn").Process(target=check)
    child.start()
    child.join(120)
    if child.is_alive():
        child.terminate()
        child.join()
        raise RuntimeError("spawn timeout")
    assert child.exitcode == 0, child.exitcode
    print("SPAWN_BOOTSTRAP_PASS", flush=True)
