$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=dlc/cond_model.bin dlc/cond_model.qnn.so
$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=dlc/sd_unet_inputs.bin dlc/sd_unet_inputs.qnn.so
$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=dlc/sd_unet_middle.bin dlc/sd_unet_middle.qnn.so
$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=dlc/sd_unet_outputs.bin dlc/sd_unet_outputs.qnn.so
$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=dlc/sd_unet_head.bin dlc/sd_unet_head.qnn.so
$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/objcopy --dump-section .autocaml.htp=dlc/decoder.bin dlc/decoder.qnn.so
