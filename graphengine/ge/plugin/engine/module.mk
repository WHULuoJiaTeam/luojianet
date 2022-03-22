
LOCAL_PATH := $(call my-dir)

COMMON_LOCAL_SRC_FILES := \
    dnnengines.cc \
    engine_manage.cc \


COMMON_LOCAL_C_INCLUDES := \
    $(LOCAL_PATH) \
    $(LOCAL_PATH)/../ \
    $(LOCAL_PATH)/../../ \
    $(TOPDIR)inc \
    $(TOPDIR)metadef/inc \
    $(TOPDIR)graphengine/inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)metadef/inc/external \
    $(TOPDIR)graphengine/inc/external \
    $(TOPDIR)graphengine/inc/framework \
    $(TOPDIR)graphengine/inc/framework/common \
	
#compiler for host libengine
include $(CLEAR_VARS)

LOCAL_SHARED_LIBRARIES := \
    libslog

LOCAL_MODULE := libengine
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DREUSE_MEMORY=1
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
endif

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES := $(COMMON_LOCAL_SRC_FILES)

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_HOST_SHARED_LIBRARY)


#compiler for device libengine
include $(CLEAR_VARS)

LOCAL_SHARED_LIBRARIES := \
    libslog

LOCAL_MODULE := libengine
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DREUSE_MEMORY=1
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
endif

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES := $(COMMON_LOCAL_SRC_FILES)

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_SHARED_LIBRARY)
