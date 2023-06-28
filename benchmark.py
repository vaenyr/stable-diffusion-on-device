import os
import re
import argparse
import logging
from pathlib import Path
from pprint import pprint

from natsort import natsorted

import autocaml.validators.snpe as snpe
import autocaml.validators.qnn as qnn


class FakeFuture():
    def __init__(self, result):
        self.r = result

    def result(self):
        return self.r

    def cancel(self):
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--regex', type=str, default=None)
parser.add_argument('--qnn', action='store_true')

args = parser.parse_args()

debug = args.debug
regex = args.regex
if regex:
    regex = re.compile(regex)

logging.basicConfig(level=logging.DEBUG-10 if debug else logging.INFO)


if args.qnn:
    bench = qnn.QnnBenchmark(devices=['S23', 'S23+', 'S23 Ultra'], runtime='htp', quantize=8, warmup=20, iters=100, detailed=True)
    file_pattern = '*.qnn.so'
    result_suffix = '.qnn.txt'
else:
    bench = snpe.SnpeBenchmark(devices=['S23', 'S23+', 'S23 Ultra'], runtime='dsp', quantize=8, warmup=20, iters=100, detailed=True)
    file_pattern = '*.int8.dlc'
    result_suffix = '.snpe.txt'

dlc_folder = Path(__file__).parent.joinpath('dlc')
results_folder = Path(__file__).parent.joinpath('results')

ok_models = []
failed_models = []
total = 0


candidates = natsorted(dlc_folder.rglob(file_pattern), key=lambda p: f'_{str(p).count(os.path.sep)}{str(p)}') # prefer less nested files first by appending a number of "/" in the path at the beginning
with bench.async_context():
    jobs = []
    for qmodel in candidates:
        if regex:
            if not regex.search(str(qmodel)):
                continue

        part = qmodel.relative_to(dlc_folder)

        print(f'Scheduling {part}... ', flush=True)
        if not args.debug:
            future = bench.run_async(qmodel, convert=False)
        else:
            future = FakeFuture(bench.run(qmodel, convert=False))
        jobs.append((part, future))

    try:
        for part, job in jobs:
            result_file = results_folder.joinpath(part)
            result_file = result_file.with_suffix(result_file.suffix + result_suffix)
            error_file = result_file.with_suffix('.error')
            result_file.parent.mkdir(parents=True, exist_ok=True)

            if result_file.exists():
                result_file.unlink()
            if error_file.exists():
                error_file.unlink()

            try:
                result = job.result()
            except Exception:
                print(part, 'Error')
                failed_models.append(part)
                import traceback
                with error_file.open('w') as f:
                    traceback.print_exc(file=f)
                    if debug:
                        traceback.print_exc()
            else:
                print(part, 'Success', result['latency'])
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
    finally:
        for _, job in jobs:
            job.cancel()
