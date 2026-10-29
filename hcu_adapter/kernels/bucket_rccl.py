"""RCCL bucket/session transport for upstream DeepEP CP collectives.

Storage aliases and session accounting follow the pinned Python protocol;
collectives use RCCL rather than unavailable symmetric-memory device APIs.
"""

import math

import torch
import torch.distributed as dist

from .deepep_dtk import EventOverlap


def get_num_allocation_alignment():
    return 2 * 1024 * 1024


def get_num_rdma_alignment():
    return 64


class BucketSession:
    def __init__(self, buffer):
        self.buffer = buffer
        self.num_allocated_bytes = 0
        self.context_idx = None

    def __enter__(self):
        if self.buffer._session is not None:
            raise RuntimeError("Nested BucketBuffer sessions are unsupported")
        self.num_allocated_bytes = 0
        self.context_idx = None
        self.buffer._session = self
        return self

    def __exit__(self, *exc):
        self.buffer._session = None

    def _check_group(self, group):
        index = self.buffer.get_context_idx(group)
        if self.context_idx not in (None, index):
            raise ValueError("All collectives in one session must share a group")
        self.context_idx = index

    def _allocate(self, num_bytes, dtype=torch.uint8):
        offset = self.num_allocated_bytes
        size = (num_bytes + 63) // 64 * 64
        if offset + size > self.buffer.num_bytes:
            raise RuntimeError("BucketBuffer session is out of storage")
        self.num_allocated_bytes += size
        return self.buffer.storage.narrow(0, offset, num_bytes).view(dtype)

    def allocate(self, shapes, dtype=torch.float32, group=None):
        self._check_group(group)
        batched = isinstance(shapes[0], (tuple, list, torch.Size))
        specs = shapes if batched else (shapes,)
        values = tuple(self._allocate(math.prod(shape) * dtype.itemsize, dtype).view(shape) for shape in specs)
        return values if batched else values[0]


class BucketBuffer:
    def __init__(self, group, allocation_plan_or_num_bytes, sl_idx=None,
                 num_gpu_timeout_secs=100, explicitly_destroy=False):
        if not isinstance(allocation_plan_or_num_bytes, int):
            raise NotImplementedError("HCU BucketBuffer currently requires an explicit byte capacity")
        if allocation_plan_or_num_bytes <= 0:
            raise ValueError("BucketBuffer capacity must be positive")
        self.num_bytes = allocation_plan_or_num_bytes
        self.groups = list(group) if isinstance(group, (tuple, list)) else [group]
        self.explicitly_destroy = explicitly_destroy
        self.storage = torch.empty(self.num_bytes, dtype=torch.uint8, device='cuda')
        self._session = None

    def get_context_idx(self, group=None):
        if group is None:
            if len(self.groups) != 1:
                raise ValueError("A group is required for a multi-group BucketBuffer")
            return 0
        return self.groups.index(group)

    def session(self):
        return BucketSession(self)

    def get_theoretical_num_sms(self, op, group=None):
        self.get_context_idx(group)
        if op not in ('all_gather', 'all_reduce', 'reduce_scatter'):
            raise ValueError("Unknown collective operation")
        return 0  # RCCL owns scheduling; no manually reserved vendor SMs.

    def _group(self, group):
        index = self.get_context_idx(group)
        if self._session is not None:
            self._session._check_group(self.groups[index])
        return self.groups[index]

    @staticmethod
    def _pack_tensors(*args):
        packed = isinstance(args[0], torch.Tensor)
        values = tuple([] if arg is None else [arg] if isinstance(arg, torch.Tensor) else list(arg) for arg in args)
        if any(len(x) not in (0, len(values[0])) for x in values):
            raise ValueError("Batched source/destination lengths differ")
        return packed, *values

    @staticmethod
    def _handle(works, outputs, packed, extra=(), scale=1.):
        handle = EventOverlap(extra_tensors=(works, outputs, extra))
        handle.num_sms = 0

        def finish():
            for work in works:
                work.wait()
            if scale != 1.:
                for output in outputs:
                    output.mul_(scale)
            return outputs[0] if packed else outputs

        handle.register_hook_after_wait(finish)
        return handle

    def all_gather(self, srcs, dsts=None, group=None, num_sms=None):
        if num_sms not in (None, 0):
            raise NotImplementedError("RCCL bucket collectives do not support custom SM reservations")
        group = self._group(group)
        packed, sources, destinations = self._pack_tensors(srcs, dsts)
        outputs, works, keep = [], [], []
        for i, source in enumerate(sources):
            source = source.contiguous()
            nbytes = source.numel() * source.element_size()
            if destinations:
                output = destinations[i]
                if output.dtype != source.dtype or not output.is_contiguous() or output.numel() != source.numel() * group.size():
                    raise ValueError("Invalid all-gather destination")
            else:
                # CPManager may supply this rank's shard of an existing session
                # allocation; recover the complete region without allocating it twice.
                offset = source.data_ptr() - self.storage.data_ptr() - group.rank() * nbytes
                if 0 <= offset and offset + nbytes * group.size() <= self.num_bytes:
                    output = self.storage.narrow(0, offset, nbytes * group.size()).view(source.dtype)
                elif self._session is not None:
                    output = self._session._allocate(nbytes * group.size(), source.dtype)
                else:
                    raise ValueError("Unregistered all-gather sources require a BucketBuffer session")
            keep.append(source)
            if nbytes:
                # Byte transport preserves FP8, BF16, packed scales and integers.
                work = dist.all_gather_into_tensor(output.view(torch.uint8).flatten(),
                                                  source.view(torch.uint8).flatten(), group=group, async_op=True)
                works.append(work)
            outputs.append(output)
        return self._handle(works, outputs, packed, keep)

    def all_reduce(self, srcs, scale=1., group=None, num_sms=None):
        if num_sms not in (None, 0):
            raise NotImplementedError("RCCL bucket collectives do not support custom SM reservations")
        group = self._group(group)
        packed, sources = self._pack_tensors(srcs)
        outputs, works = [], []
        for source in sources:
            offset = source.data_ptr() - self.storage.data_ptr()
            registered = source.is_contiguous() and 0 <= offset and offset + source.nbytes <= self.num_bytes
            if registered:
                output = source
            elif self._session is not None:
                output = self._session._allocate(source.nbytes, source.dtype).view_as(source)
                output.copy_(source)
            else:
                raise ValueError("Unregistered all-reduce sources require a session")
            works.append(dist.all_reduce(output, group=group, async_op=True))
            outputs.append(output)
        return self._handle(works, outputs, packed, sources, scale)

    def reduce_scatter(self, srcs, comm_precision='fp32', scale=1., group=None, num_sms=None):
        if num_sms not in (None, 0) or comm_precision != 'fp32':
            raise NotImplementedError("HCU bucket reduce-scatter supports default FP32 communication")
        group = self._group(group)
        packed, sources = self._pack_tensors(srcs)
        outputs, works, keep, targets = [], [], [], []
        for source in sources:
            if source.numel() % group.size():
                raise ValueError("Reduce-scatter input must divide into equal rank shards")
            offset = source.data_ptr() - self.storage.data_ptr()
            registered = source.is_contiguous() and 0 <= offset and offset + source.nbytes <= self.num_bytes
            if registered:
                region = source.flatten()
            elif self._session is not None:
                region = self._session._allocate(source.nbytes, source.dtype)
                region.copy_(source.flatten())
            else:
                raise ValueError("Unregistered reduce-scatter sources require a session")
            count = source.numel() // group.size()
            targets.append(region.narrow(0, group.rank() * count, count))
            value = source.contiguous().flatten().float()
            output = torch.empty(source.numel() // group.size(), dtype=torch.float32, device=source.device)
            works.append(dist.reduce_scatter_tensor(output, value, group=group, async_op=True))
            outputs.append(output)
            keep.append(value)
        handle = self._handle(works, outputs, packed, keep, scale)
        finish = handle.hook_after_wait

        def cast():
            result = finish()
            values = [result] if packed else result
            for target, value in zip(targets, values, strict=True):
                target.copy_(value)
            return targets[0] if packed else targets

        handle.hook_after_wait = cast
        return handle

    def barrier(self, wait_comm_stream=True, with_cpu_sync=False, sequential=True, group=None):
        work = dist.barrier(group=self._group(group), async_op=True)
        work.wait()
        if with_cpu_sync:
            torch.cuda.synchronize()

    def destroy(self):
        if self._session is not None:
            raise RuntimeError("Cannot destroy a BucketBuffer with an active session")
        self.storage = None
