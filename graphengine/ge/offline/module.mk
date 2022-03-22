
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := atc

LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DCOMPILE_OMG_PACKAGE -O2 -Dgoogle=ascend_private

LOCAL_SRC_FILES := \
    main.cc \
    single_op_parser.cc \
    ../session/omg.cc \
    ../ir_build/option_utils.cc \

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/../ ./ \
    $(TOPDIR)inc \
    $(TOPDIR)metadef/inc \
    $(TOPDIR)graphengine/inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)metadef/inc/external \
    $(TOPDIR)graphengine/inc/external \
    $(TOPDIR)metadef/inc/external/graph \
    $(TOPDIR)graphengine/inc/framework \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)metadef/inc/common/util \
    $(TOPDIR)parser    \
    third_party/json/include \
    third_party/gflags/include \
    third_party/protobuf/include \
    proto/om.proto \
    proto/ge_ir.proto \
    proto/task.proto \
    proto/insert_op.proto \

LOCAL_SHARED_LIBRARIES := \
    libc_sec \
    libge_common \
    libascend_protobuf \
    libslog \
    libgraph \
    libregister \
    liberror_manager \
    libge_compiler \
    libruntime_compile \
    libparser_common \
    liberror_manager \

LOCAL_STATIC_LIBRARIES := libgflags

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_HOST_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := atclib/atc.bin

LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DCOMPILE_OMG_PACKAGE -O2 -Dgoogle=ascend_private

LOCAL_SRC_FILES := \
    main.cc \
    single_op_parser.cc \
    ../session/omg.cc \
    ../ir_build/option_utils.cc \

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/../ ./ \
    $(TOPDIR)inc \
    $(TOPDIR)metadef/inc \
    $(TOPDIR)graphengine/inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)metadef/inc/external \
    $(TOPDIR)graphengine/inc/external \
    $(TOPDIR)metadef/inc/external/graph \
    $(TOPDIR)graphengine/inc/framework \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)metadef/inc/common/util \
    $(TOPDIR)parser    \
    third_party/json/include \
    third_party/gflags/include \
    third_party/protobuf/include \
    proto/om.proto \
    proto/ge_ir.proto \
    proto/task.proto \
    proto/insert_op.proto \

LOCAL_SHARED_LIBRARIES := \
    libc_sec \
    libge_common \
    libascend_protobuf \
    libslog \
    libgraph \
    libregister \
    liberror_manager \
    libge_compiler \
    libruntime_compile \
    libparser_common \
    liberror_manager \

LOCAL_STATIC_LIBRARIES := libgflags

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_HOST_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := fwkacl/atc.bin

LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DCOMPILE_OMG_PACKAGE -O2 -Dgoogle=ascend_private

LOCAL_SRC_FILES := \
    main.cc \
    single_op_parser.cc \
    ../session/omg.cc \
    ../ir_build/option_utils.cc \

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/../ ./ \
    $(TOPDIR)inc \
    $(TOPDIR)metadef/inc \
    $(TOPDIR)graphengine/inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)metadef/inc/external \
    $(TOPDIR)graphengine/inc/external \
    $(TOPDIR)metadef/inc/external/graph \
    $(TOPDIR)graphengine/inc/framework \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)metadef/inc/common/util \
    $(TOPDIR)parser    \
    third_party/json/include \
    third_party/gflags/include \
    third_party/protobuf/include \
    proto/om.proto \
    proto/ge_ir.proto \
    proto/task.proto \
    proto/insert_op.proto \

LOCAL_SHARED_LIBRARIES := \
    libc_sec \
    libge_common \
    libascend_protobuf \
    libslog \
    libgraph \
    libregister \
    liberror_manager \
    libge_runner \
    libruntime \
    libparser_common \
    liberror_manager \

LOCAL_STATIC_LIBRARIES := libgflags

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_HOST_EXECUTABLE)
