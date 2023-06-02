import logging
from pathlib import Path

from autocaml.validators.snpe.dlc_compiler import DLCCompiler

debug = False

logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)


dlcc = DLCCompiler()

models_folder = Path('/home/SERILOCAL/l.dudziak/dev/generative/stable-diffusion/onnx')
output_folder = Path(__file__).parent.joinpath('dlc')
output_folder.mkdir(parents=True, exist_ok=True)

with dlcc.keepfiles(debug):
    for onnx_file in models_folder.rglob('*.onnx'):
        part = onnx_file.relative_to(models_folder)

        dlc_fp32 = output_folder.joinpath(part).with_suffix('.dlc')
        dlc_int8 = dlc_fp32.with_suffix('.int8.dlc')
        dlc_fp32.parent.mkdir(parents=True, exist_ok=True)
        if not onnx_file.exists():
            print('Error: source model file does not exist!', onnx_file)
            continue

        if not dlc_fp32.exists():
            print('Attempting ONNX -> DLC (fp32) conversion for part:', part)
            try:
                dlcc.compile(onnx_file, model_type='onnx-file', output_file=dlc_fp32)
            except:
                print('Error occurred! Model will be skipped!')
                import traceback
                traceback.print_exc()
                continue

        if not dlc_int8.exists():
            print('Attempting DLC (fp32) -> DLC (int8) quantization for part:', part)
            try:
                dlcc.quantize(dlc_fp32, precision=8, output_file=dlc_int8)
            except:
                print('Error occurred! Model will be skipped!')
                import traceback
                traceback.print_exc()
                continue

        if dlc_int8.exists():
            print('Model successfully converted and quantized!', part)
        else:
            print('No error has been reported but quantized model file does not exist! Check logs.', part)
