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

ifdef LIBSD_DEBUG
APP_CPPFLAGS += -std=c++20 -O3 -Wall -Werror -fexceptions -fvisibility=hidden -DLIBSD_API="__attribute__((visibility(\"default\")))"
else
APP_CPPFLAGS += -std=c++20 -O0 -g -Wall -Werror -fexceptions
endif
APP_LDFLAGS  += -lc -lm -ldl -llog
