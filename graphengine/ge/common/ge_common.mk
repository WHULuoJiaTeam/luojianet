LOCAL_PATH := $(call my-dir)

GE_COMMON_LOCAL_SRC_FILES := \
    context/ctx.cc \
    model_saver.cc \
    ge/datatype_util.cc \
    helper/om_file_helper.cc \
    helper/model_helper.cc \
    ../model/ge_model.cc \
    ../model/ge_root_model.cc \
    auth/file_saver.cc \
    fp16_t.cc \
    math/fp16_math.cc \
    debug/memory_dumper.cc \
    formats/utils/formats_trans_utils.cc \
    dump/dump_properties.cc \
    formats/format_transfers/datatype_transfer.cc \
    formats/format_transfers/format_transfer_transpose.cc \
    formats/format_transfers/format_transfer_nchw_nc1hwc0.cc \
    formats/format_transfers/format_transfer_fractal_z.cc \
    formats/format_transfers/format_transfer_fractal_nz.cc \
    formats/format_transfers/format_transfer_fractal_zz.cc \
    formats/format_transfers/format_transfer_nhwc_nc1hwc0.cc \
    formats/format_transfers/format_transfer_nc1hwc0_nchw.cc \
    formats/format_transfers/format_transfer_nc1hwc0_nhwc.cc \
    formats/format_transfers/format_transfer_hwcn_c1hwncoc0.cc \
    formats/format_transfers/format_transfer_c1hwncoc0_hwcn.cc \
    formats/format_transfers/format_transfer_fracz_nchw.cc \
    formats/format_transfers/format_transfer_fracz_nhwc.cc \
    formats/format_transfers/format_transfer_fracz_hwcn.cc \
    formats/format_transfers/format_transfer_dhwcn_fracz3D.cc \
    formats/format_transfers/format_transfer_dhwnc_fracz3D_transpose.cc \
    formats/format_transfers/format_transfer_nchw_fz_c04.cc \
    formats/formats.cc \
    ge_format_util.cc \
    fmk_error_codes.cc \
    util.cc \
    properties_manager.cc \
    types.cc\
    model_parser/base.cc \
    kernel_store.cc \
    tbe_kernel_store.cc \
    cust_aicpu_kernel_store.cc \
    op/attr_value_util.cc \
    op/ge_op_utils.cc \
    thread_pool.cc \
    ge/tbe_plugin_manager.cc \

GE_COMMON_LOCAL_C_INCLUDES := \
    proto/om.proto \
    proto/ge_ir.proto \
    proto/task.proto \
    proto/insert_op.proto \
    proto/tensorflow/graph.proto \
    proto/tensorflow/node_def.proto \
    proto/tensorflow/function.proto \
    proto/tensorflow/versions.proto \
    proto/tensorflow/attr_value.proto \
    proto/tensorflow/tensor.proto \
    proto/tensorflow/tensor_shape.proto \
    proto/tensorflow/op_def.proto \
    proto/tensorflow/types.proto \
    proto/tensorflow/resource_handle.proto \
    $(TOPDIR)inc \
    $(TOPDIR)metadef/inc \
    $(TOPDIR)graphengine/inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)metadef/inc/external \
    $(TOPDIR)graphengine/inc/external \
    $(TOPDIR)metadef/inc/external/graph \
    $(TOPDIR)graphengine/inc/framework \
    $(TOPDIR)metadef/inc/common/util \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)third_party/json/include \
    $(TOPDIR)third_party/protobuf/include \
    $(TOPDIR)third_party/openssl/include/x86/include \
    $(TOPDIR)graphengine/ge   \
    $(TOPDIR)graphengine/ge/common   \
    $(TOPDIR)graphengine/ge/common/op   \

#compile host libge_common
include $(CLEAR_VARS)

LOCAL_MODULE := libge_common

LOCAL_CFLAGS += -Werror -DFMK_SUPPORT_DUMP -Wno-deprecated-declarations
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -O2 -Dgoogle=ascend_private

ifeq ($(DEBUG), 1)
    LOCAL_CFLAGS += -g -O0
else
    LOCAL_CFLAGS += -fvisibility=hidden -DHOST_VISIBILITY
endif
ifeq ($(host_os), euleros)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(host_os), centos)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(TARGET_OS), euleros)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(TARGET_OS), centos)
    LOCAL_CFLAGS += -DOS_CENTOS
endif

LOCAL_C_INCLUDES := $(GE_COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES := $(GE_COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libascend_protobuf \
    libc_sec \
    libslog \
    libgraph \
    libregister \
    liberror_manager \

LOCAL_STATIC_LIBRARIES += libmmpa

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_HOST_SHARED_LIBRARY)

#compile device libge_common
include $(CLEAR_VARS)

LOCAL_MODULE := libge_common

LOCAL_CFLAGS += -Werror -DFMK_SUPPORT_DUMP -Wno-deprecated-declarations
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -O2  -Dgoogle=ascend_private

ifeq ($(DEBUG), 1)
    LOCAL_CFLAGS += -g -O0
else
    LOCAL_CFLAGS += -fvisibility=hidden -DDEV_VISIBILITY
endif
ifeq ($(host_os), euleros)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(host_os), centos)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(TARGET_OS), euleros)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(TARGET_OS), centos)
    LOCAL_CFLAGS += -DOS_CENTOS
endif

LOCAL_C_INCLUDES := $(GE_COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES := $(GE_COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libascend_protobuf \
    libc_sec \
    libslog \
    libgraph \
    libregister \
    liberror_manager \

LOCAL_STATIC_LIBRARIES += libmmpa

ifeq ($(device_os),android)
LOCAL_LDFLAGS += -ldl
LOCAL_LDLIBS += -L$(PWD)/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29 -llog
else
LOCAL_LDFLAGS := -lrt -ldl
endif

include $(BUILD_SHARED_LIBRARY)

#compile host libge_common static lib
include $(CLEAR_VARS)

LOCAL_MODULE := libge_common

LOCAL_CFLAGS += -Werror -DFMK_SUPPORT_DUMP -Wno-deprecated-declarations
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -O2  -Dgoogle=ascend_private

ifeq ($(DEBUG), 1)
    LOCAL_CFLAGS += -g -O0
endif

ifeq ($(host_os), euleros)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(host_os), centos)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(TARGET_OS), euleros)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(TARGET_OS), centos)
    LOCAL_CFLAGS += -DOS_CENTOS
endif

LOCAL_C_INCLUDES := $(GE_COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES := $(GE_COMMON_LOCAL_SRC_FILES)

LOCAL_STATIC_LIBRARIES := \
    libgraph     \
    libascend_protobuf  \

LOCAL_SHARED_LIBRARIES := \
    libc_sec \
    libslog \
    libmmpa \
    libregister \
    liberror_manager \

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_HOST_STATIC_LIBRARY)

#compile device libge_common static_lib
include $(CLEAR_VARS)

LOCAL_MODULE := libge_common

LOCAL_CFLAGS += -Werror -DFMK_SUPPORT_DUMP -Wno-deprecated-declarations
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -O2  -Dgoogle=ascend_private

ifeq ($(DEBUG), 1)
    LOCAL_CFLAGS += -g -O0
endif
ifeq ($(host_os), euleros)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(host_os), centos)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(TARGET_OS), euleros)
    LOCAL_CFLAGS += -DOS_CENTOS
endif
ifeq ($(TARGET_OS), centos)
    LOCAL_CFLAGS += -DOS_CENTOS
endif

LOCAL_C_INCLUDES := $(GE_COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES := $(GE_COMMON_LOCAL_SRC_FILES)

LOCAL_STATIC_LIBRARIES := \
    libgraph     \
    libascend_protobuf  \

LOCAL_SHARED_LIBRARIES := \
    libc_sec \
    libslog \
    libmmpa \
    libregister \
    liberror_manager \

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_STATIC_LIBRARY)
