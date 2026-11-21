# SPDX-License-Identifier: Apache-2.0
# Core033 schemas; #69057 packing is semantic, not an owner ABI extension.
SCHEMAS = (
    (
        'hisparse_resolve_residency(Tensor host_cache, '
        'Tensor! hot_cache, '
        'Tensor hot_block_table, '
        'Tensor global_indices, '
        'Tensor! hot_indices, '
        'Tensor! device_global_indices, '
        'Tensor! lru_slots, '
        'Tensor? request_state_indices, '
        'int region_stride, '
        'int max_union_rows, '
        'Tensor(a!)? miss_mask=None, '
        'Tensor(b!)? stats=None, '
        'Tensor(c!)? attention_indices=None, '
        'int attention_block_stride=0, '
        'Tensor? request_ids=None, '
        'Tensor? source_block_table=None, '
        'int source_block_size=0, '
        'Tensor(d!)? resolved_global_indices=None, '
        'Tensor(e!)? valid_counts=None, '
        'Tensor(f!)? swap_host_physical_rows=None, '
        'Tensor(g!)? swap_device_physical_rows=None, '
        'Tensor(h!)? swap_counts=None, '
        'Tensor? resident_block_table=None, '
        'int resident_block_size=0, '
        'int resident_null_block=0) -> ()'
    ),
    (
        'hisparse_invalidate_written_slots(Tensor! device_global_indices, '
        'Tensor request_state_indices, '
        'Tensor req_id_per_token, '
        'Tensor written_slots) -> ()'
    ),
    (
        'hisparse_gather_plan(Tensor host_cache, '
        'Tensor! hot_cache, '
        'Tensor global_indices, '
        'Tensor hot_indices, '
        'Tensor miss_mask, '
        'Tensor? request_state_indices, '
        'Tensor(a!)? attention_indices=None, '
        'int attention_block_stride=0) -> ()'
    ),
    (
        'hisparse_gather_compact(Tensor host_cache, '
        'Tensor! hot_cache, '
        'Tensor miss_global_indices, '
        'Tensor miss_hot_indices, '
        'Tensor miss_counts) -> ()'
    ),
)
