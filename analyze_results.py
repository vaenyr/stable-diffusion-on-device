import os
import re
import argparse
import logging
from pathlib import Path
from tabulate import tabulate

from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--regex', type=str, default=None)

args = parser.parse_args()

debug = args.debug
regex = args.regex
if regex:
    regex = re.compile(regex)

logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
results_folder = Path(__file__).parent.joinpath('results')

ok_models = []
failed_models = []
candidates = natsorted(results_folder.rglob('*.txt'), key=lambda p: f'_{str(p).count(os.path.sep)}{str(p)}')
for results in candidates:
    if regex:
        if not regex.search(str(results)):
            continue

    content = results.read_text()
    name, result = content.split('\n', maxsplit=1)

    result = eval(result)['layers']
    results = list(result.items())
    results = sorted(results, key=lambda p: p[1]['latency'], reverse=True)
    results = [(r[1]['name'], r[1]['latency']) for r in results[:10]]
    print(name)
    print(tabulate(results, headers=['Layer', 'Latency']))
    print()

