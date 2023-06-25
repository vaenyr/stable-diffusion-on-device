# ==============================================================================
#
#  Copyright (c) 2020, 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ===============================================================

APP_ABI      := arm64-v8a armeabi-v7a
APP_STL      := c++_shared
APP_PLATFORM := android-21
APP_CPPFLAGS += -std=c++21 -O3 -Wall -Werror -fvisibility=hidden -DQNN_API="__attribute__((visibility(\"default\")))" -DLIBSD_API="__attribute__((visibility(\"default\")))"
APP_LDFLAGS  += -nodefaultlibs -lc -lm -ldl -lgcc -llog
