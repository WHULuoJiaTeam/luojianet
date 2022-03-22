LOCAL_PATH := $(call my-dir)

# task.proto is old task, add it for ops_kernel_info_store
local_ge_runtime_src_files :=         \
    model_runner.cc                   \
    runtime_model.cc                  \
    output.cc                         \
    task/aicpu_task.cc                \
    task/cce_task.cc                  \
    task/tbe_task.cc                  \
    task/event_record_task.cc         \
    task/event_wait_task.cc           \
    task/stream_active_task.cc        \
    task/stream_switch_task.cc        \
    task/hccl_task.cc                 \
    task/memcpy_async_task.cc         \
    task/profiler_task.cc             \

local_ge_runtime_include :=           \
    $(LOCAL_PATH)/                    \
    $(TOPDIR)libc_sec/include         \
    $(TOPDIR)inc/external             \
    $(TOPDIR)inc/external/graph       \
    $(TOPDIR)inc/framework            \
    $(TOPDIR)inc/graph                \
    $(TOPDIR)inc                      \
    $(LOCAL_PATH)/../                 \
    third_party/protobuf/include

local_ge_runtime_shared_library :=    \
    libruntime                        \
    libslog                           \
    libc_sec

local_ge_runtime_ldflags := -lrt -ldl

# compile device libge_runtime
include $(CLEAR_VARS)

LOCAL_MODULE := libge_runtime
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -O2
LOCAL_CFLAGS += -Werror
LOCAL_SRC_FILES := $(local_ge_runtime_src_files)
LOCAL_C_INCLUDES := $(local_ge_runtime_include)
LOCAL_SHARED_LIBRARIES := $(local_ge_runtime_shared_library)
LOCAL_LDFLAGS += $(local_ge_runtime_ldflags)

include $(BUILD_SHARED_LIBRARY)

# compile host libge_runtime
include $(CLEAR_VARS)

LOCAL_MODULE := libge_runtime
LOCAL_CFLAGS += -Werror
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0
ifeq ($(DEBUG), 1)
    LOCAL_CFLAGS += -g -O0
else
    LOCAL_CFLAGS += -O2
endif
LOCAL_SRC_FILES := $(local_ge_runtime_src_files)
LOCAL_C_INCLUDES := $(local_ge_runtime_include)
LOCAL_SHARED_LIBRARIES := $(local_ge_runtime_shared_library)
LOCAL_LDFLAGS += $(local_ge_runtime_ldflags)

include $(BUILD_HOST_SHARED_LIBRARY)
