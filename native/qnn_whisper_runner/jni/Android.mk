LOCAL_PATH := $(call my-dir)

ifndef QNN_SDK_ROOT
$(error QNN_SDK_ROOT must point to an extracted QAIRT SDK)
endif

QNN_SAMPLE_ROOT := $(QNN_SDK_ROOT)/examples/QNN/SampleApp/SampleApp/src

include $(CLEAR_VARS)
LOCAL_MODULE := qnn-whisper-runner
LOCAL_C_INCLUDES := \
    $(QNN_SDK_ROOT)/include/QNN \
    $(QNN_SAMPLE_ROOT) \
    $(QNN_SAMPLE_ROOT)/Log \
    $(QNN_SAMPLE_ROOT)/PAL/include \
    $(QNN_SAMPLE_ROOT)/Utils \
    $(QNN_SAMPLE_ROOT)/WrapperUtils
LOCAL_SRC_FILES := \
    persistent_runner.cpp \
    $(QNN_SAMPLE_ROOT)/QnnSampleApp.cpp \
    $(QNN_SAMPLE_ROOT)/Log/Logger.cpp \
    $(QNN_SAMPLE_ROOT)/Log/LogUtils.cpp \
    $(QNN_SAMPLE_ROOT)/PAL/src/linux/Directory.cpp \
    $(QNN_SAMPLE_ROOT)/PAL/src/linux/DynamicLoading.cpp \
    $(QNN_SAMPLE_ROOT)/PAL/src/linux/FileOp.cpp \
    $(QNN_SAMPLE_ROOT)/PAL/src/linux/Path.cpp \
    $(QNN_SAMPLE_ROOT)/PAL/src/common/GetOpt.cpp \
    $(QNN_SAMPLE_ROOT)/PAL/src/common/StringOp.cpp \
    $(QNN_SAMPLE_ROOT)/Utils/DataUtil.cpp \
    $(QNN_SAMPLE_ROOT)/Utils/DynamicLoadUtil.cpp \
    $(QNN_SAMPLE_ROOT)/Utils/IOTensor.cpp \
    $(QNN_SAMPLE_ROOT)/Utils/QnnDlcUtils.cpp \
    $(QNN_SAMPLE_ROOT)/Utils/QnnSampleAppUtils.cpp \
    $(QNN_SAMPLE_ROOT)/WrapperUtils/QnnWrapperUtils.cpp
LOCAL_LDLIBS := -ldl -llog
include $(BUILD_EXECUTABLE)
