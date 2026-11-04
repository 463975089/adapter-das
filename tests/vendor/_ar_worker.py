# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Distributed aiter custom-all-reduce stress worker (HCU adapter).

Run by tests/vendor/test_allreduce.py via torchrun (>=2 ranks), with
HCU_AITER_AR=1 so the adapter's gfx93* refuse-gate is bypassed on purpose:
we are testing the vendor kernel itself. Selection goes through sglang's
production dispatch (dispatch_custom_allreduce), so what fails here is
what production would run.

Checks per (size, dtype):
  eager   aiter AR vs exact fp32 gold sum (all_gather'd, fp32-accumulated);
  nccl    same check for NCCL (harness sanity: if both fail, harness bug);
  graph   capture AR into a CUDA graph, replay 4x with FRESH inputs —
          output must track the new inputs. This is the exact detector for
          the gfx93x replay-corruption class (1eb038d): eager can pass
          while every replayed result is garbage or frozen.

Prints one "ARSTRESS" line per check; exits nonzero on any FAIL.
"""

import os

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

import torch
import torch.distributed as dist


def msg(rank, kind, name, detail=""):
    line = f"ARSTRESS {kind} {name}" + (f" :: {detail}" if detail else "")
    if rank == 0:
        print(line, flush=True)
    return kind


def main():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    fails = []

    dist.init_process_group("nccl")
    # production parity (parallel_state): CA attaches to a gloo cpu group,
    # never the NCCL world group.
    cpu_group = dist.new_group(backend="gloo")
    try:
        # newer sglang builds read process-global ServerArgs during dispatch
        # (e.g. runtime_context); set a minimal dummy so the production
        # selection path actually runs instead of erroring out.
        try:
            from sglang.srt.server_args import ServerArgs

            # ServerArgs validates model_path as an HF repo id unless the
            # dir exists; fabricate a minimal plain (non-quantized) config
            # so construction succeeds regardless of what's on disk.
            path = os.environ.get("HCU_AR_STRESS_MODEL") or \
                "/tmp/hcu_ar_stress_model"
            os.makedirs(path, exist_ok=True)
            if not os.path.exists(os.path.join(path, "config.json")):
                import json as _json

                with open(os.path.join(path, "config.json"), "w") as _f:
                    _json.dump(
                        {"model_type": "llama",
                         "architectures": ["LlamaForCausalLM"],
                         "hidden_size": 512, "num_hidden_layers": 2,
                         "num_attention_heads": 8, "vocab_size": 32000},
                        _f,
                    )
            sa = ServerArgs(model_path=path)
            try:
                from sglang.srt.runtime_context import _CONTEXT

                _CONTEXT.set_server_args(sa)
            except Exception:
                pass
            try:
                from sglang.srt.server_args import (
                    set_global_server_args_for_scheduler as _set_sa,
                )

                _set_sa(sa)
            except Exception:
                pass
        except Exception as e:
            if rank == 0:
                print(f"ARSTRESS INFO server-args init skipped: {e!r:.120}",
                      flush=True)

        import sglang.srt.distributed.device_communicators.\
            custom_all_reduce as car

        ar = car.dispatch_custom_allreduce(group=cpu_group, device=dev)
    except Exception as e:  # gate active, wheel missing, init fail, ...
        msg(rank, "SKIP", "dispatch", repr(e)[:300])
        dist.destroy_process_group()
        raise SystemExit(0)

    if ar is None:
        msg(rank, "SKIP", "dispatch", "dispatch returned None (no aiter AR)")
        dist.destroy_process_group()
        raise SystemExit(0)

    # dispatch returns either an AR instance (older sglang: .all_reduce()) or
    # a constructor partial (newer sglang: caller supplies group/device,
    # mirroring parallel_state's CAClass(group=cpu_group, device=...)).
    if hasattr(ar, "all_reduce") or hasattr(ar, "custom_all_reduce"):
        inst = ar
    else:
        inst = ar(group=cpu_group, device=dev)
        dist.barrier()
    # prefer custom_all_reduce (returns None when the size is out of the
    # kernel's range — production then falls back); else raw all_reduce.
    run = getattr(inst, "custom_all_reduce", None) or \
        getattr(inst, "all_reduce", None)
    if not callable(run):
        msg(rank, "SKIP", "api", f"{type(inst).__name__} has no AR entry")
        dist.destroy_process_group()
        raise SystemExit(0)
    print(f"ARSTRESS INFO backend={type(inst).__name__} "
          f"entry={'custom_all_reduce' if hasattr(inst, 'custom_all_reduce') else 'all_reduce'}",
          flush=True)

    SIZES = [8192, 65536, 262144, 1048576, 8388608]  # elements
    TOL = {
        torch.bfloat16: 3e-2,
        torch.float16: 6e-3,
        torch.float32: 1e-4,
    }

    def gold(x):
        buf = [torch.empty_like(x, dtype=torch.float32) for _ in range(world)]
        dist.all_gather(buf, x.float())
        return torch.stack(buf).sum(0)

    def close(a, g, tol):
        scale = g.abs().max().item() + 1e-6
        return bool(((a.float() - g).abs() <= tol * scale).all())

    for nelem in SIZES:
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            name = f"{nelem}x{str(dt).split('.')[-1]}"
            try:
                torch.manual_seed(1234 + nelem)
                x = (torch.randn(nelem, device=dev) * 3).to(dt)

                # --- eager aiter AR vs gold (None = size out of range) ---
                yr = run(x.clone())
                dist.barrier()
                if yr is None:
                    msg(rank, "SKIP", name + ":range",
                        "should_custom_ar=False (out of vendor range)")
                    continue
                g0 = gold(x)
                if not close(yr, g0, TOL[dt]):
                    fails.append(name + ":eager")
                    msg(rank, "FAIL", name + ":eager",
                        f"max|d|={(yr.float()-g0).abs().max().item():.4e}")

                # --- NCCL harness sanity ---
                yn = x.clone()
                dist.all_reduce(yn)
                if not close(yn, g0, TOL[dt]):
                    fails.append(name + ":nccl-ref")
                    msg(rank, "FAIL", name + ":nccl-ref", "harness suspect")

                # --- graph capture + replay with fresh inputs ---
                static = x.clone()
                for _ in range(3):
                    run(static)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = run(static)
                ok = True
                for trial in range(4):
                    fresh = (torch.randn(nelem, device=dev) * 3).to(dt)
                    static.copy_(fresh)
                    graph.replay()
                    torch.cuda.synchronize()
                    gt = gold(fresh)
                    res = out if out is not None else static
                    if not close(res, gt, TOL[dt]):
                        ok = False
                        d = (res.float() - gt).abs().max().item()
                        msg(rank, "FAIL",
                            f"{name}:graph-replay[{trial}]",
                            f"max|d|={d:.4e} (does not track fresh input)")
                        break
                if ok:
                    msg(rank, "PASS", name)
                else:
                    fails.append(name + ":graph")
                dist.barrier()
            except Exception as e:
                fails.append(name + ":error")
                msg(rank, "ERROR", name, repr(e)[:300])
                dist.barrier()

    if rank == 0:
        if fails:
            print(f"ARSTRESS SUMMARY: FAIL {fails}", flush=True)
            code = 1
        else:
            print("ARSTRESS SUMMARY: PASS", flush=True)
            code = 0
    else:
        code = 0
    dist.destroy_process_group()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
