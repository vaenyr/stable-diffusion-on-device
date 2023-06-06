import os
import re
import argparse
import logging
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait

from natsort import natsorted

from autocaml.validators.snpe.dlc_compiler import DLCCompiler

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--regex', type=str, default=None)
parser.add_argument('--force', action='store_true')
parser.add_argument('--path', type=str, default=str(Path(__file__).absolute().parents[1].joinpath('stable-diffusion', 'onnx')))

args = parser.parse_args()

debug = args.debug
regex = args.regex
if regex:
    regex = re.compile(regex)

logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)


dlcc = DLCCompiler()

models_folder = Path(args.path)
output_folder = Path(__file__).parent.joinpath('dlc')
output_folder.mkdir(parents=True, exist_ok=True)


def convert_onnx(onnx_file):
    part = onnx_file.relative_to(models_folder)
    dlc_fp32 = output_folder.joinpath(part).with_suffix('.dlc')
    dlc_int8 = dlc_fp32.with_suffix('.int8.dlc')
    dlc_fp32.parent.mkdir(parents=True, exist_ok=True)
    if not onnx_file.exists():
        print('Error: source model file does not exist!', onnx_file)
        return

    if not dlc_fp32.exists() or args.force:
        print('Attempting ONNX -> DLC (fp32) conversion for part:', part)
        try:
            dlcc.compile(onnx_file, model_type='onnx-file', output_file=dlc_fp32)
        except Exception:
            print('Error occurred while converting! Model will be skipped!', part)
            import traceback
            traceback.print_exc()
            return

    print('waiting...')
    input()
    if not dlc_int8.exists() or args.force:
        print('Attempting DLC (fp32) -> DLC (int8) quantization for part:', part)
        try:
            dlcc.quantize(dlc_fp32, precision=8, output_file=dlc_int8)
        except Exception:
            print('Error occurred while quantizing! Model will be skipped!', part)
            import traceback
            traceback.print_exc()
            return

    if dlc_int8.exists():
        print('Model successfully converted and quantized!', part)
    else:
        print('No error has been reported but quantized model file does not exist! Check logs.', part)


with dlcc.keepfiles(debug):
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        try:
            candidates = natsorted(models_folder.rglob('*.onnx'), key=lambda p: f'_{str(p).count(os.path.sep)}{str(p)}')
            jobs = []
            for onnx_file in candidates:
                if regex:
                    if not regex.search(str(onnx_file)):
                        continue

                #jobs.append(pool.submit(convert_onnx, onnx_file))
                convert_onnx(onnx_file)

            wait(jobs)
        finally:
            for job in jobs:
                job.cancel()
            pool.shutdown(wait=False)
