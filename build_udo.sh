#!/bin/bash

function usage() {
    echo 'Usage:'
    echo '"./build_udo.sh" build incrementally'
    echo '"./build_udo.sh rebuild" clean and then build from scratch'
    echo '"./build_udo.sh clean" clean but do not build'
}

clean=0
run=1

while [[ $# -gt 0 ]]; do
  case $1 in
    rebuild)
      clean=1
      shift # past argument
      ;;
    clean)
      clean=1
      run=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      usage
      exit 1
      ;;
  esac
done


if [[ $clean == 1 ]]; then
    if [ -e "$SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/core.py.bak" ]; then
        mv "$SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/core.py.bak" "$SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/core.py"
    fi

    if [ -e "$SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/op_factory.py.bak" ]; then
        mv "$SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/op_factory.py.bak" "$SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/op_factory.py"
    fi

    if [ -e sdod/csrc/GroupNormPackage ]; then
        rm -rf sdod/csrc/GroupNormPackage
    fi
fi

if [[ $run == 1 ]]; then
    if [ ! -e "$SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/core.py.bak" ]; then
        echo "Applying snpe_custom_op_core.patch..."
        cp $SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/core.py $SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/core.py.bak
        patch $SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/core.py patches/snpe_custom_op_core.patch
    fi

    if [ ! -e "$SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/op_factory.py.bak" ]; then
        echo "Applying snpe_custom_op_factory.patch..."
        cp $SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/op_factory.py $SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/op_factory.py.bak
        patch $SNPE_ROOT/lib/python/qti/aisw/converters/backend/custom_ops/op_factory.py patches/snpe_custom_op_factory.patch
    fi

    SNPE_UDO_ROOT=$SNPE_ROOT/share/SnpeUdo \
        LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH:$(dirname $SNPE_PYTHON)/../lib \
        PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH \
        $SNPE_PYTHON $(which snpe-udo-package-generator) -p config/group_norm.json -o sdod/csrc/

    # fix for gentoo...
    is_utils_linux_rename=$(rename --help | grep -o "<replacement>")
    if [ ! -z "$is_utils_linux_rename" ] && [ ! -e sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile.bak ]; then
        echo "Patching Makefile to handle util-linux rename..."
        cp sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile.bak
        sed -i "s+'s/arm64-v8a/aarch64-android/'+arm64-v8a aarch64-android+" sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile
        sed -i "s+'s/armeabi-v7a/arm-android/'+armeabi-v7a arm-android+" sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile
        sed -iE 's/rename_target_dirs \(.*\) \\/rename_target_dirs \1 ; \\/' sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile
        sed -iE 's/&& find \(.*\)\\/find \1 || true/' sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile
    fi

    cp sdod/csrc/GroupNormCpuImpl.cpp sdod/csrc/GroupNormPackage/jni/src/CPU/src/ops/GroupNorm.cpp
    cp sdod/csrc/GroupNormCpuValidation.cpp sdod/csrc/GroupNormPackage/jni/src/reg/GroupNormPackageCpuImplValidationFunctions.cpp
    cp sdod/csrc/GroupNormDspV73Impl.cpp sdod/csrc/GroupNormPackage/jni/src/DSP_V73/GroupNormImplLibDsp.cpp
    cp sdod/csrc/GroupNormDspV73Validation.cpp sdod/csrc/GroupNormPackage/jni/src/reg/GroupNormPackageDspImplValidationFunctions.cpp


    pushd sdod/csrc/GroupNormPackage >/dev/null 2>&1 && {
        make cpu_x86
    popd >/dev/null 2>&1
    }
fi
