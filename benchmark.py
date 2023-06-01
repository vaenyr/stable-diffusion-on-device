import logging
from pathlib import Path
from pprint import pprint

import autocaml.validators.snpe as snpe

debug = False

logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)


bench = snpe.SnpeBenchmark(devices='S23+', runtime='dsp', quantize=8)

models_folder = Path('/home/SERILOCAL/l.dudziak/dev/generative/stable-diffusion/onnx')
parts = [
    'sd_unet_inputs.onnx',
    'sd_unet_middle.onnx',
    'sd_unet_outputs.onnx',
    'sd_unet_head.onnx',
]

for part in parts:
    qmodel = (models_folder / part).with_suffix('.int8.dlc')
    try:
        result = bench.run(qmodel, convert=False)
    except:
        print('Error occurred while benchmarking!')
        import traceback
        traceback.print_exc()
    else:
        print(part)
        pprint(result)
        print()
