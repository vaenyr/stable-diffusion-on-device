# Stable-Diffusion On-device

This repo contains a collections of components used to run Stable Diffusion On-Device.
Things are still work-in-progress, but ultimately a demo Android app should be available to be build.

Components (current, future and tentative):
- conversion scripts to obtain QNN/SNPE models from ONNX/Torchscript
    - TODO: collect data for quantization and do it properly, currently using random inputs...
    - TODO[low prio]: test TorchScript -> ONNX path and see if it improves, perhaps export full UNet in a single checkpoint? But we have to keep in mind RAM spikes when loading models...
- scripts to benchmark and analyze performance of QNN/SNPE models
- Custom OPs (TBD if we need any and what benefits they give):
    - python package to expose custom ops (and various other PyTorch utilities): `sdod`
    - QNN custom operations package: `csrc/sdod_ops`
- C/C++ library to drive the image generation process on-device: `csrc/libsdod`
- Android application `java/sdod_app` (TODO, at the end)

See the remaining sections for info about each component.

## Converting and benchmarking models

Requirements:

- Modified Stable-Diffusion repo: https://github.sec.samsung.net/SAIC-Cambridge/stable-diffusion
    - follow its README to setup the base environment for running the original model
    - the rest of this documentation assumes the default conda environment called `ldm` has been created during the above step, please make necessary adjustments if needed
- Working QNN setup:
    - QNN 2.10 (other versions not tested), unpacked to a directory called `<qnn_path>`, QNN can be downloaded from Qualcomm CreatePoint
    - Python 3.6 environment with QNN/SNPE dependencies, if you use conda you can use `environment.yml` provided with this repo, this will create an environment called `snpe`
        - NOTE: you do not have to have the environment active, this environment is not the same as the `ldm` environment
        - the rest of the documentation assumes a conda environment called `snpe` is used, located at `<snpe_env_path>` (e.g., `~/miniconda3/envs/snpe`), adjust if needed
    - set the following environment variables:
        - `export QNN_SDK_ROOT="<qnn_path>"`
        - `export QNN_PYTHON="<snpe_env_path>/bin/python"`
- Android NDK and host clang:
    - for clang, simply install it using your OS package manager, e.g., on Ubuntu: `sudo apt install clang`
        - **Note:** while building the QNN models works with pretty much any clang (clang 6.0 on Ubuntu 18 does the job), we need a more recent one to build the C/C++ library in the following steps; if you want to build the library, you should have clang that recognizes `-std=c++20` switch (14+ seems to be ok)
    - for NDK, we assume no previous installation of Android Studio/SDK, please follow the steps below; if you have an SDK, simply make sure the environment variables are set and the correct NDK version is used:
        - create a folder to hold your Android SDK, located at `<android_sdk_path>`, e.g., `~/android/sdk`
        - set environment variables `export ANDROID_HOME="<android_sdk_path>"`
        - follow the steps outlined at the beginning of: https://developer.android.com/tools/sdkmanager to get command line tools
            - after that, you should have the following structure in your SDK folder:
            ```
                <android_sdk_path>
                    └── cmdline-tools
                        └── latest
            ```
        - install the recent NDK bundle (r22): `$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager --install ndk-bundle`
            - similar to the host clang situation, we recommend more recent NDK (r22 or above) to make sure you don't face problems when compiling the C++ library, if you only want to run individual models, you can probably go with an older one
            - if you are behind proxies, add a variation of the following: `--no_https --proxy=http --proxy_host=<proxy_addr> --proxy_port=<proxy_port>`
            - you can optionally install a side-by-side version of an ndk, e.g.: `$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager --install --install "ndk;25.2.9519653"`
        - set the environment variable NDK_HOME to point to the desired version of the NDK, e.g., if using the ndk bundle:
            - `export NDK_HOME="<android_sdk_path>/ndk-bundle"
        - expose NDK and SDK tools in your PATH:
            - `export PATH="$PATH:$NDK_HOME:$ANDROID_HOME/platform-tools:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/tools:$ANDROID_HOME/tools/bin"`
- `autocaml` package from `dev/diffusion` branch: https://github.sec.samsung.net/SAIC-Cambridge/autocaml/tree/dev/diffusion
    - recommended to install it inside the `ldm` environment, but you can install it in a separate one; however, do not install it in the `snpe` environment!

Steps to follow:

0. We assume you can run the original model following the stable-diffusion repo, e.g., running the following command inside the `ldm` environment should give you a valid picture under `<sd_repo>/outputs/txt2img-samples/samples`:
    - `(ldm) user@host:<sd_repo>$ python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms --n_iter=1 --skip_grid --ddim_steps=20 --n_samples=1`
    - if `--plms` is unrecognized, try `--sampler=plms`, you can also try DPM solver with `--sampler=dpm`, see: https://github.sec.samsung.net/SAIC-Cambridge/stable-diffusion/pull/1
1. Install this repo in the `ldm` environment: `(ldm) user@host:<sdod_repo>$ pip install -e .`
2. Export optimized version of the stable diffusion model to ONNX (note: you can also try unoptimized version by removing the `--optimize` flag):
    `(ldm) user@host:<sd_repo>$ ... --onnx --scale=1 --optimize`
3. A folder `<sd_repo>/onnx` should be created with different parts of the models created
        - to correctly export the CLIP encoder, extra changes need to done in the transformers library; I'ill try to share the details later, for now if anything related to `cond_model` or `cond_stage_model` fails simply try to ignore/comment it
4. Navigate to the stable-diffusion-on-device repo
5. call conversion script:
    - `(ldm) user@host:<sdod_repo>$ python todlc.py --force --qnn --path <sd_repo>/onnx`
    - this will attempt to create all `*.onnx` files from the given folder and generate corresponding `*.qnn.so` files under `<sdod_repo>/dlc`, mirroring the folder structure
    - note: if the SD and SDOD repos share the same parent folder, the `--path` argument can be skipped, the same hold true for other similar cases
    - note: some models might fail, but most of them should convert ok
    - note: you can also convert only a subset of models by passing `--regex REGEX` option, where `REGEX` is a regular expression searched for in each file's path: if it is found, the file will be included for conversion
    - note: you can add `--debug` for detailed output, but we only recommend doing so when converting a single model (`--regex` pointint to a particular file); with `--debug` all intermediate files are also kept for the user to investigate, look for messages: `New temporary directory: <path>` and navigate accordingly
6. benchmark converted models:
    - make sure you have at least one S23 available in adb, i.e., `adb devices` should output at least one device
        - if no devices are available, or none of them is recognized as S23/S23+/S23 Ultra, you will an error `RuntimeError: No devices available`
        - if you want to try benchmarking on devices other than S23, modify `devices=['S23', 'S23+', 'S23 Ultra']` in `benchmark.py` according to your needs
        - if more than one devices is available, benchmarking of different models will be parallelized
        - note: the script assumes exclusive access to the devices, it uses `/data/tmp/local/autocaml` directory to store all necessary files
    - `(ldm) user@host:<sdod_repo>$ python benchmark.py --qnn`
        - note: you can also use `--regex` option to only benchmark selected models
        - note: `--debug` is also available for verbose output
        - note: by default detailed (per-layer) profiling is used, this adds some overhead and can be disabled if desired by changing `detailed=True` arguments in the script
    - running the command above will create a folder `<sdod_repo>/results` which again mirrors the `dlc/` folder -> each source model will have a `.txt` file with full profiling results if running a model succeeded, or `.error` file with exception information in case of failure
7. (optional) analyze results by running `(ldm) user@host:<sdod_repo>$ python analyze_results.py --qnn [--regex REGEX]`, this will print the most time-consuming operations for each model (requires `detailed=True` in the benchmarking script)


### (Optional) Use custom operations

TODO

## Building the C/C++ library

Requirements:

- the setup from the previous part (converting and benchmarking models) should be enough, provided recent versions of clang and ndk have been installed
    - clang-14 and clang-16 have been tested and seem to work fine
    - ndk-22 and ndk-25 have been tested and seem to work fine

Steps to follow:

1. extract HTP binary blobs from `*.qnn.so` files:
    1. navigate to `<sdod_repo>/dlc`
    1. run the following commands:
    ```
        user@host:<sdod_repo>/dlc$ $NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=cond_model.bin cond_model.qnn.so
        user@host:<sdod_repo>/dlc$ $NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=decoder.bin decoder.qnn.so
        user@host:<sdod_repo>/dlc$ $NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=sd_unet_inputs.bin sd_unet_inputs.qnn.so
        user@host:<sdod_repo>/dlc$ $NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=sd_unet_middle.bin sd_unet_middle.qnn.so
        user@host:<sdod_repo>/dlc$ $NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=sd_unet_outputs.bin sd_unet_outputs.qnn.so
        user@host:<sdod_repo>/dlc$ $NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=sd_unet_head.bin sd_unet_head.qnn.so
    ```
2. build the library
    1. go to `<sdod_repo>/csrc/libsdod`
    2. run `make aarch64-android` to build for Android, or `make all_x86` to build for host (useful for debugging)
        - note: you can also prepend `LIBSDOD_DEBUG=1` to any of the make commands to get a debug build (but be careful as debug and release builds share the same artifacts names)
        - note: the script does not clean things by default, if a fresh build is desired, run `make clean <normal_target>`
3. (optional) test the library on the host machine
    0. this step requires building the library for `all_x86` target
    1. copy necessary QNN libraries: `cp $QNN_SDK_ROOT/target/x86_64-linux-clang/lib/libQnnHtp.so $QNN_SDK_ROOT/target/x86_64-linux-clang/lib/libQnnSystem.so <sdod_repo>/csrc/libsdod/bin/x86_64-linux-clang/`
    2. navigate to the folder holding binaries: `cd <sdod_repo>/csrc/libsdod/bin/x86_64-linux-clang/`
    3. run the testing app: `host@user:<sdod_repo>/csrc/libsdod/bin/x86_64-linux-clang$ LB_LIBRARY_PATH=$(pwd) ./test`
        - note: graph execution on the host might take very long time, since it simulates execution on an actual HTP HW; if you reach this point, feel free to interrupt the program
4. test the library on the phone
    0. this step requires building the library for `aarch64-android` target
    1. copy necessary QNN libraries: `cp $QNN_SDK_ROOT/target/aarch64-android/lib/libQnnHtp.so $QNN_SDK_ROOT/target/aarch64-android/lib/libQnnSystem.so $QNN_SDK_ROOT/target/aarch64-android/lib/libQnnHtpV73Stub.so $QNN_SDK_ROOT/target/hexagon-v73/lib/unsigned/libQnnHtpV73Skel.so bin/aarch64-android/`
    2. create target folder on-device: `adb shell mkdir -p /data/local/tmp/libsdod`
    3. push binary blobs: `adb push <sdod_repo>/dlc/*.bin /data/local/tmp/libsdod`
    4. push binaries: `adb push <sdod_repo>/csrc/libsdod/bin/aarch64-android/* /data/local/tmp/libsdod`
    5. run the testing app: `adb shell cd /data/local/tmp/libsdod "&&" 'LD_LIBRARY_PATH=$(pwd)' ./test`

Library status:
- QNN setup and teardown: DONE
- Loading models: DONE
- Match profiling speed: DONE
- zero-copy data allocation (Android only): DONE
- Correctly connect inputs/outputs of different models: DONE
- Handle (de)quantization of input/output data: DONE
- Handle interpolation between conditional and unconditional outputs: DONE
- CLIP tokenizer: TODO
- DPM solver: TODO
