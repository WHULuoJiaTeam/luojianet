
LOCAL_PATH := $(call my-dir)

COMMON_LOCAL_SRC_FILES := \
    proto/ge_api.proto \
    ge_api.cc \


COMMON_LOCAL_C_INCLUDES := \
    proto/ge_ir.proto \
    proto/task.proto \
    proto/om.proto \
    proto/insert_op.proto \
    $(LOCAL_PATH) ./ \
    $(LOCAL_PATH)/../ \
    $(LOCAL_PATH)/../../ \
    $(TOPDIR)inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)inc/external/graph \
    $(TOPDIR)inc/common \
    $(TOPDIR)inc/framework \
    $(TOPDIR)inc/graph \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)ops/built-in/op_proto/inc \
    third_party/json/include \
    third_party/protobuf/include \
    third_party/opencv/include \

DEVICE_LOCAL_C_INCLUDES := \
    proto/ge_ir.proto \
    proto/task.proto \
    proto/om.proto \
    proto/insert_op.proto \
    $(LOCAL_PATH) ./ \
    $(LOCAL_PATH)/../ \
    $(LOCAL_PATH)/../../ \
    $(TOPDIR)inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)inc/external/graph \
    $(TOPDIR)inc/framework \
    $(TOPDIR)inc/common \
    $(TOPDIR)inc/graph \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)ops/built-in/op_proto/inc \
    third_party/json/include \
    third_party/protobuf/include \
    third_party/opencv/include \

#compiler for host infer
include $(CLEAR_VARS)

LOCAL_MODULE := libge_client
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DREUSE_MEMORY=1 -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
endif

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec \
    libascend_protobuf \
    libslog \
    libmmpa \
    libgraph \
    libregister \
    libge_compiler \
    libge_common

LOCAL_STATIC_LIBRARIES += libmsprofiler_fwk \


LOCAL_LDFLAGS := -lrt -ldl

LOCAL_SHARED_LIBRARIES += \
    libruntime \

include $(BUILD_HOST_SHARED_LIBRARY)

#compiler for device
include $(CLEAR_VARS)

LOCAL_MODULE := libge_client
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -DGOOGLE_PROTOBUF_NO_RTTI -DDEV_VISIBILITY
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0
LOCAL_CFLAGS += -DOMG_DEVICE_VERSION  -DREUSE_MEMORY=1  -Dgoogle=ascend_private
LOCAL_MODULE_CLASS := SHARED_LIBRARIES

LOCAL_C_INCLUDES := $(DEVICE_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec \
    libascend_protobuf \
    libslog \
    libmmpa \
    libgraph \
    libregister \
    libruntime \
    libge_compiler \
    libge_common


LOCAL_STATIC_LIBRARIES += libmsprofiler_fwk \


LOCAL_LDFLAGS := -lrt -ldl
LOCAL_CFLAGS += \
    -Wall

include $(BUILD_SHARED_LIBRARY)
