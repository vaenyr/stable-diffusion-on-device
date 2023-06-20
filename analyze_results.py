import os
import re
import sys
import argparse
import logging
from pathlib import Path
from tabulate import tabulate

from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--regex', type=str, default=None)
parser.add_argument('--qnn', action='store_true')
parser.add_argument('--op_summary', action='store_true')

args = parser.parse_args()


def determine_op_type(name):
    if args.qnn:
        name = name.replace('__', '_')
        name = name.replace('_', '/')

    if name == '/0/op/Conv':
        return 'conv'
    if name.startswith('gelu_'):
        return 'act'
    if name.startswith('/0/in_layers/in_layers.0/'):
        return 'norm'
    if name.startswith('/0/in_layers/in_layers.1/'):
        return 'act'
    if name.startswith('/0/in_layers/in_layers.2/'):
        return 'conv'
    if name.startswith('/0/emb_layers/emb_layers.0/'):
        return 'act'
    if name.startswith('/0/emb_layers/emb_layers.1/'):
        return 'matmul'
    if name.startswith('/0/out_layers/out_layers.0/'):
        return 'norm'
    if name.startswith('/0/out_layers/out_layers.1/'):
        return 'act'
    if name.startswith('/0/out_layers/out_layers.2/'):
        return 'dropout'
    if name.startswith('/0/out_layers/out_layers.3/'):
        return 'conv'
    if name.startswith('/0/skip_connection/'):
        return 'skip-conn'
    if name.startswith('/0/Add'):
        return 'skip-conn'
    if name.startswith('layernorm_'):
        return 'norm'
    if name.startswith('/1/Add'):
        return 'skip-conn'
    if name.startswith('/1/proj_in/'):
        return 'conv'
    if name.startswith('/1/norm/'):
        return 'norm'
    if name.startswith('/1/proj_out/'):
        return 'conv'
    if name.startswith('/1/transformer_blocks.0/norm1/'):
        return 'norm'
    if name.startswith('/1/transformer_blocks.0/norm2/'):
        return 'norm'
    if name.startswith('/1/transformer_blocks.0/norm3/'):
        return 'norm'
    if name.startswith('/1/transformer_blocks.0/Add'):
        return 'skip-conn'
    if name.startswith('/1/transformer_blocks.0/attn'):
        if '/to_out/to_out.0/' in name:
            return 'matmul'
        if '/to_out/to_out.1/' in name:
            return 'dropout'
        if '/to_' in name:
            return 'matmul'
        if '/smax/' in name:
            return 'softmax'
        if '/MatMul' in name:
            return 'matmul'
    if name.startswith('/1/transformer_blocks.0/ff/net/net.0/proj/'):
        return 'matmul'
    if name.startswith('/1/transformer_blocks.0/ff/net/net.0/'):
        return 'act'
    if name.startswith('/1/transformer_blocks.0/ff/net/net.1/'):
        return 'dropout'
    if name.startswith('/1/transformer_blocks.0/ff/net/net.2/'):
        return 'matmul'

    if 'Reshape' in name or 'Transpose' in name or 'Unsqueeze' in name:
        return 'shaping'

    print('Unknown operation!', name, 'Falling back to "other"', file=sys.stderr)
    return 'other'

debug = args.debug
regex = args.regex
if regex:
    regex = re.compile(regex)

logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
results_folder = Path(__file__).parent.joinpath('results')

latency_by_type = {}
total_by_type = 0
total_latency = 0
total_ops = 0

if args.qnn:
    pattern = '*.qnn.txt'
else:
    pattern = '*.snpe.txt'

ok_models = []
failed_models = []
candidates = natsorted(results_folder.rglob(pattern), key=lambda p: f'_{str(p).count(os.path.sep)}{str(p)}')
for results in candidates:
    if regex:
        if not regex.search(str(results)):
            continue

    content = results.read_text()
    name, result = content.split('\n', maxsplit=1)

    result = eval(result)
    part_latency = result['latency']
    total_latency += part_latency

    result = result['layers']
    results = list(result.items())
    results = sorted(results, key=lambda p: p[1]['latency'], reverse=True)
    results = [(r[1]['name'], r[1]['latency']) for r in results]
    print(name)
    print(tabulate(results[:10], headers=['Layer', 'Latency']))
    print()

    for name, latency in results:
        total_ops += latency
        if args.op_summary:
            t = determine_op_type(name)
            latency_by_type.setdefault(t, 0)
            latency_by_type[t] += latency
            total_by_type += latency

if args.op_summary:
    latency_by_type = sorted([(key, value, round(value/total_by_type*100, 2), round(value/total_by_type * total_latency, 4)) for key, value in latency_by_type.items()], key=lambda p: p[1], reverse=True)
    print(tabulate(latency_by_type, headers=['Op.', 'Cycles', '% Cycles', 'Approx. latency']))
print('Total latency of operations:', total_ops)
print('Total latency (ms):', total_latency)
