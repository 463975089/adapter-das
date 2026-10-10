"""Distributed aiter custom-all-reduce stress worker (HCU adapter).

Run by tests/test_vendor_allreduce.py via torchrun (>=2 ranks), with
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
    try:
        import sglang.srt.distributed.device_communicators.\
            custom_all_reduce as car

        ar = car.dispatch_custom_allreduce(dist.group.WORLD, dev)
    except Exception as e:  # gate active, wheel missing, init fail, ...
        msg(rank, "SKIP", "dispatch", repr(e)[:300])
        dist.destroy_process_group()
        raise SystemExit(0)

    if ar is None:
        msg(rank, "SKIP", "dispatch", "dispatch returned None (no aiter AR)")
        dist.destroy_process_group()
        raise SystemExit(0)

    run = getattr(ar, "all_reduce", None)
    if not callable(run):
        msg(rank, "SKIP", "api", f"{type(ar).__name__} has no all_reduce()")
        dist.destroy_process_group()
        raise SystemExit(0)

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

                # --- eager aiter AR vs gold ---
                yr = run(x.clone())
                dist.barrier()
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
