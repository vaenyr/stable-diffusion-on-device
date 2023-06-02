import os
import re
import argparse
import logging
from pathlib import Path
from pprint import pprint
from functools import cmp_to_key

from natsort import natsorted

import autocaml.validators.snpe as snpe

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--regex', type=str, default=None)
parser.add_argument('--force', action='store_true')

args = parser.parse_args()

debug = args.debug
regex = args.regex
if regex:
    regex = re.compile(regex)

logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)


bench = snpe.SnpeBenchmark(devices='S23+', runtime='dsp', quantize=8)

dlc_folder = Path(__file__).parent.joinpath('dlc')
results_folder = Path(__file__).parent.joinpath('results')

ok_models = []
failed_models = []
total = 0
candidates = natsorted(dlc_folder.rglob('*.int8.dlc'), key=lambda p: f'_{str(p).count(os.path.sep)}{str(p)}') # prefer less nested files first by appending a number of "/" in the path at the beginning
for qmodel in candidates:
    if regex:
        if not regex.search(str(qmodel)):
            continue

    part = qmodel.relative_to(dlc_folder)
    result_file = results_folder.joinpath(part).with_suffix('.txt')
    error_file = result_file.with_suffix('.error')
    result_file.parent.mkdir(parents=True, exist_ok=True)

    if result_file.exists():
        result_file.unlink()
    if error_file.exists():
        error_file.unlink()

    try:
        print(f'Running {part}... ', end='', flush=True)
        result = bench.run(qmodel, convert=False)
    except:
        print('Error')
        failed_models.append(part)
        import traceback
        with error_file.open('w') as f:
            traceback.print_exc(file=f)
    else:
        print('Success', result['latency'])
        ok_models.append(part)
        total += result['latency']
        with result_file.open('w') as f:
            print(part, file=f)
            pprint(result, stream=f)
            print(file=f)

if failed_models:
    print('Failed models:')
    for p in failed_models:
        print('   ', p)
print('Successes:', len(ok_models), 'Failures:', len(failed_models))
if len(ok_models) > 1:
    print('Total latency of successful models:', total)
