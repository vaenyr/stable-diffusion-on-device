import logging
from pathlib import Path
from pprint import pprint

import autocaml.validators.snpe as snpe

debug = False

logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)


bench = snpe.SnpeBenchmark(devices='S23 Ultra', runtime='dsp', quantize=8)

dlc_folder = Path(__file__).parent.joinpath('dlc')
results_folder = Path(__file__).parent.joinpath('results')

ok_models = []
failed_models = []
for qmodel in dlc_folder.rglob('*.int8.dlc'):
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
        with result_file.open('w') as f:
            print(part, file=f)
            pprint(result, stream=f)
            print(file=f)

print('Failed models:')
for p in failed_models:
    print('   ', p)
print('Successes:', len(ok_models), 'Failures:', len(failed_models))
