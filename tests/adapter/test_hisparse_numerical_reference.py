# SPDX-License-Identifier: Apache-2.0
"""Known answers for the independent future HIP test oracle, WITHOUT Torch/GPU."""
import ast
from copy import deepcopy
from pathlib import Path


def _oracle():
    path = Path(__file__).resolve().parents[1] / 'kernels/hisparse/test_native.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('_case', '_resolve_reference')]
    namespace = {'deepcopy': deepcopy}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_case'], namespace['_resolve_reference']


def _run(name):
    case, resolve = _oracle()
    config = case(name)
    stats = [0, 0]
    result = resolve(config['indices'], config['hot_table'], config['dgi'], config['lru'],
                     host_rows=64, requests=config['requests'], states=config['states'],
                     source=config['source'], resident=config['resident'], block_size=2,
                     attention_stride=5, stats=stats)
    return config, result, stats


def test_independent_numerical_oracle_known_hole_and_eviction():
    config, out, stats = _run('holes_eviction')
    assert out['hot'] == [[-1, 6, 2, 2]]
    assert out['attention'] == [[15, 5, 5, -1]] and out['counts'] == [3]
    assert out['mask'] == [[0, 0, 1, 0]] and out['swaps'] == [1]
    assert out['sh'] == [[14, -77, -77, -77]]
    assert config['dgi'] == [[14, 11, 12, 13]]
    assert config['lru'] == [[1, 3, 0, 2]] and stats == [2, 1]


def test_independent_numerical_oracle_resident_and_padding():
    config, out, stats = _run('resident_only')
    assert out['hot'] == [[-1, 4, -1, 5]]
    assert out['attention'] == [[10, 11, -1, -1]] and out['counts'] == [2]
    assert config['dgi'] == [[10, 11, 12, 13]] and stats == [0, 0]
    config, out, stats = _run('padding')
    assert out['hot'] == [[-1] * 4] and out['counts'] == [0]
    assert out['mask'] == [[0] * 4] and out['swaps'] == [0]


def test_independent_numerical_oracle_multiple_request_states():
    config, out, stats = _run('multi_request')
    assert out['hot'] == [[-1, 8, 12, 12], [13, 9, -1, 12], [-1, 6, 2, 2]]
    assert out['swaps'] == [2, 0, 1] and out['counts'] == [3, 3, 3]
    assert out['sh'][0][:2] == [22, 23] and out['sh'][2][0] == 14
    assert config['lru'] == [[2, 3, 0, 1], [1, 3, 0, 2]]
    assert stats == [6, 3]
