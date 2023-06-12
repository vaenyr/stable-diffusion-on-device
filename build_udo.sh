#!/bin/bash

SNPE_UDO_ROOT=$SNPE_ROOT/share/SnpeUdo \
    LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH:$(dirname $SNPE_PYTHON)/../lib \
    PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH \
    $SNPE_PYTHON $(which snpe-udo-package-generator) -p config/group_norm.json -o sdod/csrc/ -f

cp sdod/csrc/GroupNormCpuImpl.cpp sdod/csrc/GroupNormPackage/jni/src/CPU/src/ops/GroupNorm.cpp
cp sdod/csrc/GroupNormCpuValidation.cpp sdod/csrc/GroupNormPackage/jni/src/reg/GroupNormPackageCpuImplValidationFunctions.cpp
cp sdod/csrc/GroupNormDspV73Impl.cpp sdod/csrc/GroupNormPackage/jni/src/DSP_V73/GroupNormImplLibDsp.cpp
cp sdod/csrc/GroupNormDspV73Validation.cpp sdod/csrc/GroupNormPackage/jni/src/reg/GroupNormPackageDspImplValidationFunctions.cpp


# fix for gentoo...
is_utils_linux_rename=$(rename --help | grep -o "<replacement>")
if [ ! -z "$is_utils_linux_rename" ]; then
    echo "Patching Makefile to handle util-linux rename..."
    cp sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile.bak
    sed -i "s+'s/arm64-v8a/aarch64-android/'+arm64-v8a aarch64-android+" sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile
    sed -i "s+'s/armeabi-v7a/arm-android/'+armeabi-v7a arm-android+" sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile
    sed -iE 's/rename_target_dirs \(.*\) \\/rename_target_dirs \1 ; \\/' sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile
    sed -iE 's/&& find \(.*\)\\/find \1 || true/' sdod/csrc/GroupNormPackage/jni/src/CPU/Makefile
fi

pushd sdod/csrc/GroupNormPackage >/dev/null 2>&1 && {
    make cpu
popd >/dev/null 2>&1
}

