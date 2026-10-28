"""HCU DeepEP v1 communication with the pinned v2 expanded-token contract.

Numerical transforms live here, independent of deepinfer. Only an explicitly
selected, single-host baseline is supported. Graph providers are separate
opt-in candidates and require fixed-capacity dispatch without CPU sync.
"""

from dataclasses import dataclass
import socket

import torch
import torch.distributed as dist


class EventOverlap:
    """Keep vendor events and payloads alive until the deferred epilogue runs."""

    def __init__(self, event=None, extra_tensors=None):
        self.event = event
        self.extra_tensors = extra_tensors
        self.hook_after_wait = None
        self._release = False

    def register_hook_after_wait(self, hook):
        if self.hook_after_wait is not None:
            raise RuntimeError("An event epilogue is already registered")
        self.hook_after_wait = hook

    def current_stream_wait(self, release_handle=False):
        if self.event is not None:
            self.event.current_stream_wait()
        result = None
        if self.hook_after_wait is not None:
            hook, self.hook_after_wait = self.hook_after_wait, None
            result = hook()
        if release_handle:
            self.event = self.extra_tensors = None
        return result

    wait = current_stream_wait

    def __call__(self, release_handle=False):
        self._release = release_handle
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.current_stream_wait(self._release)
        self._release = False


@dataclass
class EPHandle:
    vendor_handle: object
    psum_num_recv_tokens_per_expert: torch.Tensor
    expanded_to_received: torch.Tensor
    num_recv_tokens: int
    vendor_buffer: object = None


def _bytes(x):
    return x.contiguous().view(torch.uint8)


def _pack(x):
    if not isinstance(x, tuple):
        if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % 128:
            raise ValueError("BF16 dispatch requires a matrix with hidden divisible by 128")
        return x.contiguous(), None
    value, scale = x
    if (value.dtype != torch.float8_e4m3fn or value.ndim != 2 or scale.ndim != 2
            or scale.shape[0] != value.shape[0]
            or scale.dtype not in (torch.float32, torch.int32)):
        raise ValueError("FP8 dispatch requires data[M,K] and float/int32 scales[M,S]")
    width = value.shape[1] + scale.shape[1] * scale.element_size()
    padded = (width + 255) // 256 * 256
    storage = torch.zeros((value.shape[0], padded), dtype=torch.uint8, device=value.device)
    storage[:, :value.shape[1]].copy_(_bytes(value))
    storage[:, value.shape[1]:width].copy_(_bytes(scale))
    return storage.view(torch.bfloat16), (value.shape[1], scale.shape[1], scale.dtype)


def _unpack(x, spec, col_major):
    if spec is None:
        return x
    hidden, groups, dtype = spec
    raw = _bytes(x)
    value = raw[:, :hidden].contiguous().view(torch.float8_e4m3fn)
    scale = raw[:, hidden:hidden + groups * 4].contiguous().view(dtype)
    if col_major:
        rows = (x.shape[0] + 127) // 128 * 128
        aligned = torch.empty((groups, rows), dtype=dtype, device=x.device).T[:x.shape[0]]
        aligned.copy_(scale)
        scale = aligned
    return value, scale


def _expand(x, indices, weights, counts, experts, alignment):
    """Sort valid expert slots, pad each expert, retain a reverse token map."""
    rows, topk = indices.shape
    valid = (indices >= 0) & (indices < experts)
    ids = torch.where(valid, indices, experts).flatten()
    order = torch.argsort(ids, stable=True)
    sorted_ids = ids[order]
    histogram = torch.zeros(experts + 1, dtype=torch.int32, device=x.device)
    histogram.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    actual = histogram[:experts]
    aligned = (actual + alignment - 1) // alignment * alignment
    starts = torch.cat((aligned.new_zeros(1), aligned.cumsum(0)))
    raw_starts = torch.cat((actual.new_zeros(1), actual.cumsum(0)))
    # CPU counts are already returned by the vendor in eager mode. In the
    # no-sync path allocate a static bound; prefix sums still mark valid data.
    capacity = (sum((n + alignment - 1) // alignment * alignment for n in counts)
                if counts else rows * topk + experts * (alignment - 1))
    position = (starts[sorted_ids] + torch.arange(order.numel(), device=x.device)
                - raw_starts[sorted_ids])
    position = torch.where(sorted_ids < experts, position, capacity).long()
    expanded = x.new_zeros((capacity + 1, x.shape[1]))
    expanded.index_copy_(0, position, x.index_select(0, order // topk))
    recv_map = torch.full((capacity + 1,), rows, dtype=torch.int64, device=x.device)
    recv_map.index_copy_(0, position, order // topk)
    expanded_weights = weights.new_zeros(capacity + 1)
    expanded_weights.index_copy_(0, position, weights.flatten()[order])
    # v2 expand psum stores each expert's actual end; next start is aligned.
    psum = (starts[:-1] + actual).to(torch.int32)
    return expanded[:-1], expanded_weights[:-1], recv_map[:-1], psum


def get_physical_domain_size(group_or_buffer):
    """Pinned v2 contract: (rdma ranks, LSA ranks), LSA = ncclTeamLsa size.

    LSA means symmetric-memory load/store access, not raw P2P. On DTK2610 the
    raw intra-host P2P matrix is all-pairs, but RCCL exports no LSA team/device
    pointer API and VMM reservation is capped (R-16), so no peer is LSA.
    """
    context = getattr(group_or_buffer, "context", None)
    if context is not None:
        return context.get_physical_domain_size()
    return group_or_buffer.size(), 1


def get_logical_domain_size(group_or_buffer, allow_hybrid_mode=None):
    """Pinned v2 contract: (scale-out ranks, scale-up ranks)."""
    context = getattr(group_or_buffer, "context", None)
    if context is not None:
        return context.get_logical_domain_size()
    num_rdma_ranks, num_lsa_ranks = get_physical_domain_size(group_or_buffer)
    if allow_hybrid_mode is None or allow_hybrid_mode:
        return num_rdma_ranks, num_lsa_ranks
    return 1, num_rdma_ranks * num_lsa_ranks


class EPBuffer:
    """Explicit v1 Buffer bridge; all numerical dispatch uses the installed wheel."""

    vendor_type = None
    graph_capture_enabled = False

    @staticmethod
    def get_buffer_size_hint(group, num_max_tokens_per_rank=0, hidden=0, num_topk=0,
                             use_fp8_dispatch=False, **kwargs):
        # Use the vendor's actual channel workspace sizing. Include packed SF
        # transport width and the routing metadata in the conservative width.
        vendor = EPBuffer.vendor_type
        width = (hidden * 2 + num_topk * 16 + 255) // 256 * 256
        hints = [getattr(vendor, name)(group.size()).get_nvl_buffer_size_hint(width, group.size())
                 for name in ("get_dispatch_config", "get_combine_config")]
        size = max(*hints, 32 << 20, group.size() * num_max_tokens_per_rank
                   * (hidden * 2 + num_topk * 16) * 4)
        return (size + (1 << 20) - 1) // (1 << 20) * (1 << 20)

    def __init__(self, group, num_max_tokens_per_rank=0, hidden=0, num_topk=0,
                 use_fp8_dispatch=False, allow_hybrid_mode=True,
                 allow_multiple_reduction=True, explicitly_destroy=False, **kwargs):
        if kwargs or not allow_multiple_reduction:
            raise NotImplementedError("HCU v1 EP bridge supports default multiple-reduction only")
        if group.size() > 8:
            raise NotImplementedError("HCU EP bridge is currently limited to one host, <=8 ranks")
        hosts = [None] * group.size()
        dist.all_gather_object(hosts, socket.gethostname(), group=group)
        if len(set(hosts)) != 1:
            raise NotImplementedError("HCU EP bridge is currently limited to one host")
        self.group = group
        self.explicitly_destroy = explicitly_destroy
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_bytes = self.get_buffer_size_hint(group, num_max_tokens_per_rank, hidden, num_topk)
        self.vendor = self.vendor_type(group, num_nvl_bytes=self.num_bytes,
                                       explicitly_destroy=explicitly_destroy)

    def get_theoretical_num_sms(self, num_experts, num_topk):
        return self.vendor_type.num_sms

    def capture(self):
        return self.vendor.capture().event

    def destroy(self):
        self.vendor.destroy()

    def _dispatch_vendor(self, limit, hidden, topk, do_cpu_sync):
        return self.vendor

    def dispatch(self, x, topk_idx=None, topk_weights=None,
                 cumulative_local_expert_recv_stats=None, num_experts=None,
                 num_max_tokens_per_rank=None, expert_alignment=1, num_sms=0, num_qps=0,
                 previous_event=None, async_with_compute_stream=False,
                 allocate_on_comm_stream=False, handle=None, do_handle_copy=False,
                 do_cpu_sync=None, do_expand=False, do_zero_padding=False,
                 use_tma_aligned_col_major_sf=False, defer_epilogue=False):
        if handle is not None or do_handle_copy or num_qps or do_zero_padding:
            raise NotImplementedError("Cached EP dispatch, custom QPs and zero-padding mode are not qualified")
        if not do_expand:
            raise NotImplementedError("The baseline EP bridge implements expanded dispatch only")
        if num_sms not in (0, self.vendor_type.num_sms):
            raise NotImplementedError("Use the installed Buffer.num_sms for HCU communication")
        if torch.cuda.is_current_stream_capturing():
            if not self.graph_capture_enabled:
                raise NotImplementedError("EP graph capture requires the explicit GraphEPBuffer candidate")
            if do_cpu_sync is not False:
                raise ValueError("EP graph capture requires do_cpu_sync=False and a fixed receive bound")
        limit = num_max_tokens_per_rank or self.num_max_tokens_per_rank
        if limit <= 0:
            raise ValueError("EP dispatch requires a positive token bound")
        data, spec = _pack(x)
        if data.shape[0] > limit or num_experts % self.group.size():
            raise ValueError("Dispatch exceeds configured token bound or has uneven expert shards")
        # FP8 transport packs into a narrower BF16 matrix, but combine returns
        # full-width BF16 activations. Size the workspace for both directions.
        hidden = max(data.shape[1], spec[0]) if spec is not None else data.shape[1]
        vendor = self._dispatch_vendor(limit, hidden, topk_idx.shape[1], do_cpu_sync)
        event = None if previous_event is None else EventOverlap(previous_event)
        per_rank, per_rdma, per_expert, rank_mask, layout_event = vendor.get_dispatch_layout(
            topk_idx, num_experts, previous_event=event, async_finish=True)
        result = vendor.dispatch(
            data, topk_idx=topk_idx, topk_weights=topk_weights,
            num_tokens_per_rank=per_rank, num_tokens_per_rdma_rank=per_rdma,
            num_tokens_per_expert=per_expert, is_token_in_rank=rank_mask,
            expert_alignment=1, num_worst_tokens=0 if do_cpu_sync is not False else limit * self.group.size(),
            previous_event=layout_event, async_finish=True)
        recv, idx, weight, counts, vendor_handle, vendor_event = result
        overlap = EventOverlap(vendor_event.event, (result, data, layout_event))

        def finish():
            local_experts = num_experts // self.group.size()
            expanded, expanded_weight, mapping, psum = _expand(
                recv, idx, weight, counts, local_experts, expert_alignment)
            if cumulative_local_expert_recv_stats is not None:
                actual = torch.zeros(local_experts + 1, dtype=torch.int32, device=idx.device)
                ids = torch.where((idx >= 0) & (idx < local_experts), idx, local_experts).flatten()
                actual.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
                cumulative_local_expert_recv_stats.add_(actual[:-1])
            ep_handle = EPHandle(vendor_handle, psum, mapping, recv.shape[0], vendor)
            return _unpack(expanded, spec, use_tma_aligned_col_major_sf), None, expanded_weight, ep_handle

        overlap.register_hook_after_wait(finish)
        if defer_epilogue:
            return overlap
        return *overlap.current_stream_wait(), overlap

    def combine(self, x, handle, topk_weights=None, bias=None, num_sms=0, num_qps=0,
                previous_event=None, async_with_compute_stream=False,
                allocate_on_comm_stream=False, defer_epilogue=False):
        if num_qps or num_sms not in (0, self.vendor_type.num_sms):
            raise NotImplementedError("Custom EP communication configuration is not qualified")
        if previous_event is not None:
            previous_event.current_stream_wait()
        value = x.float()
        if topk_weights is not None:
            value = value * topk_weights[:, None]
        folded = torch.zeros((handle.num_recv_tokens + 1, x.shape[1]),
                             device=x.device, dtype=torch.float32)
        folded.index_add_(0, handle.expanded_to_received, value)
        vendor = handle.vendor_buffer if handle.vendor_buffer is not None else self.vendor
        result = vendor.combine(folded[:-1].to(torch.bfloat16), handle.vendor_handle,
                                bias=bias, async_finish=True)
        out, weights, vendor_event = result
        overlap = EventOverlap(vendor_event.event, (result, folded))
        overlap.register_hook_after_wait(lambda: (out, weights))
        if defer_epilogue:
            return overlap
        return *overlap.current_stream_wait(), overlap


class EngramBuffer:
    """GPU sharded lookup using RCCL collectives, with byte-exact payloads.

    This baseline implements lookup results, not the v2 GDAKI transport or QP
    tuning. Tables stay sharded. Only query-sized buffers are communicated.
    CPU RDMA tables are unsupported. Graph capture requires GraphEngramBuffer.
    """

    graph_capture_enabled = False

    def __init__(self, group, num_gpu_bytes, num_rdma_storage_size,
                 use_cpu_rdma_storage, num_allocated_qps, qp_depth,
                 restrict_rd_atomic=False, allow_hybrid_mode=True,
                 explicitly_destroy=False):
        if use_cpu_rdma_storage:
            raise NotImplementedError("Engram RCCL candidate needs DEEPINFER_ENGRAM_FORCE_GPU_STORAGE=1")
        self.group = group
        self.rank_idx, self.num_ranks = group.rank(), group.size()
        self.num_gpu_bytes = num_gpu_bytes
        self.num_rdma_storage_bytes = num_rdma_storage_size
        self.storages = self.sfs = None
        self.config = None

    @staticmethod
    def get_storage_size_hint(group, num_entries_per_layer, hidden, num_max_tokens,
                              num_entries_per_token, dtype=torch.bfloat16, num_sf_packs=0):
        query_bytes = (group.size() * num_max_tokens * num_entries_per_token
                       * (hidden * dtype.itemsize + 8) * len(num_entries_per_layer))
        sf_bytes = sum(num_entries_per_layer) * group.size() * num_sf_packs * 4
        table_bytes = sum(num_entries_per_layer) * hidden * dtype.itemsize
        return query_bytes + sf_bytes, table_bytes

    @staticmethod
    def get_theoretical_config(group, num_layers, num_max_tokens, num_entries_per_token):
        # No RDMA QPs are allocated by this transport. These compatibility
        # placeholders are not performance recommendations for DeepEP v2.
        return 1, 1, False

    def set_config(self, num_entries_per_layer, hidden, num_max_tokens,
                   num_entries_per_token, dtype=torch.bfloat16, num_sf_packs=0):
        if self.storages is not None:
            raise RuntimeError("Engram configuration cannot change after write")
        if dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise TypeError("Engram candidate supports BF16/FP8 tables only")
        self.config = (tuple(num_entries_per_layer), hidden, num_max_tokens,
                       num_entries_per_token, dtype, num_sf_packs)

    def write(self, storages, sfs=None):
        if self.config is None or self.storages is not None:
            raise RuntimeError("Configure Engram exactly once before write")
        entries, hidden, _, _, dtype, packs = self.config
        if len(storages) != len(entries) or (packs > 0) != (sfs is not None):
            raise ValueError("Engram layer/scale count does not match configuration")
        if sfs is not None and len(sfs) != len(entries):
            raise ValueError("Engram scaling-factor layer count mismatch")
        for i, (storage, rows) in enumerate(zip(storages, entries, strict=True)):
            if not storage.is_cuda or storage.shape != (rows, hidden) or storage.dtype != dtype:
                raise ValueError("Engram tables must be real GPU tensors matching configuration")
            if sfs is not None and (sfs[i].device != storage.device
                                    or sfs[i].shape != (rows * self.num_ranks, packs)
                                    or sfs[i].dtype != torch.int32):
                raise ValueError("Engram SF must be replicated packed int32 table")
        # Own storage independently of the model loader's subsequent release.
        self.storages = [storage.clone() for storage in storages]
        self.sfs = None if sfs is None else [sf.clone() for sf in sfs]

    def fetch(self, indices, num_qps=0, use_tma_aligned_col_major_sf=True):
        if self.storages is None:
            raise RuntimeError("Engram write must complete before fetch")
        if num_qps not in (0, 1):
            raise NotImplementedError("Engram candidate supports default transport only")
        if torch.cuda.is_current_stream_capturing() and not self.graph_capture_enabled:
            raise NotImplementedError("Engram graph capture requires the explicit GraphEngramBuffer candidate")
        entries, hidden, max_tokens, per_token, dtype, packs = self.config
        layers, tokens, width = indices.shape
        if layers != len(entries) or tokens > max_tokens or width != per_token:
            raise ValueError("Engram query exceeds configured layer/token/entry dimensions")
        if indices.dtype not in (torch.int32, torch.int64) or indices.device != self.storages[0].device:
            raise ValueError("Engram indices must be int32/int64 on the table GPU")
        padded = indices.new_full((layers, max_tokens, per_token), -1)
        padded[:, :tokens].copy_(indices)
        global_queries = indices.new_empty((self.num_ranks * layers, max_tokens, per_token))
        dist.all_gather_into_tensor(global_queries, padded, group=self.group)
        global_queries = global_queries.view(self.num_ranks, layers, max_tokens, per_token)
        hooks = []
        for layer, (rows, storage) in enumerate(zip(entries, self.storages, strict=True)):
            query = global_queries[:, layer].contiguous().flatten()
            local = query - self.rank_idx * rows
            selected = _bytes(storage).index_select(0, local.clamp(0, rows - 1).long())
            payload = torch.where(((local >= 0) & (local < rows))[:, None], selected, 0)
            work = dist.all_reduce(payload, op=dist.ReduceOp.MAX, group=self.group, async_op=True)
            sf = None
            if packs:
                local_query = indices[layer].flatten().long()
                sf = self.sfs[layer].index_select(0, local_query.clamp(0, rows * self.num_ranks - 1))
                sf = torch.where(((local_query >= 0) & (local_query < rows * self.num_ranks))[:, None], sf, 0)
                sf = sf.reshape(tokens, per_token * packs)
                if use_tma_aligned_col_major_sf:
                    stride = (tokens + 127) // 128 * 128
                    col = torch.empty((sf.shape[1], stride), dtype=sf.dtype, device=sf.device).T[:tokens]
                    col.copy_(sf)
                    sf = col

            def wait(payload=payload, work=work, sf=sf):
                work.wait()
                result = payload.view(self.num_ranks, max_tokens, per_token * hidden * dtype.itemsize)
                result = result[self.rank_idx, :tokens].view(dtype)
                return result if sf is None else (result, sf)

            hooks.append(wait)
        return hooks

    def destroy(self):
        self.storages = self.sfs = self.config = None


class GraphEPBuffer(EPBuffer):
    """Separate fixed-capacity graph transport from CPU-sized prefill transport.

    DTK v1 notify_dispatch writes mapped host receive counters even in a graph.
    A queued replay can overwrite the CPU reset for a later dynamic dispatch,
    causing that dispatch to allocate from stale counts (attempt42). Dedicated
    buffers isolate those counters without adding a host synchronization.
    """

    graph_capture_enabled = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_vendor = None
        self.graph_num_bytes = 0

    def _dispatch_vendor(self, limit, hidden, topk, do_cpu_sync):
        if do_cpu_sync is not False:
            return self.vendor
        size = self.get_buffer_size_hint(self.group, limit, hidden, topk)
        if self.graph_vendor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up fixed-capacity EP dispatch on all ranks before capture")
            # The first (largest) decode warmup sizes this small workspace;
            # duplicating the prefill-sized workspace would waste GiBs.
            self.graph_vendor = self.vendor_type(
                self.group, num_nvl_bytes=size, explicitly_destroy=self.explicitly_destroy)
            self.graph_num_bytes = size
        if size > self.graph_num_bytes:
            raise ValueError("Fixed-capacity EP exceeds its warmup workspace; warm up the largest shape first")
        return self.graph_vendor

    def destroy(self):
        if self.graph_vendor is not None:
            self.graph_vendor.destroy()
            self.graph_vendor = None
        super().destroy()


class GraphEngramBuffer(EngramBuffer):
    """Opt-in GPU-table RCCL graph candidate; queries change in device buffers."""

    graph_capture_enabled = True
