if (ENABLE_OPEN_SRC)
    if(DEFINED ENV{D_LINK_PATH})
        # D_LINK_PATH is set
        set(GE_LIB_PATH $ENV{D_LINK_PATH})
        set(GE_SYS_ARCH "")
        if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
            # x86 ubuntu
            set(GE_SYS_ARCH "x86_64")
        elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
            # arm euleros
            set(GE_SYS_ARCH "aarch64")
        else()
            message(FATAL_ERROR "Running on a unsupported architecture: ${SYSTEM_TYPE}, build terminated")
        endif()
        set(GE_LIB_PATH ${GE_LIB_PATH}/${GE_SYS_ARCH})
        find_module(slog libalog.so ${GE_LIB_PATH})
        find_module(static_mmpa libmmpa.a ${GE_LIB_PATH})
    elseif (DEFINED ENV{ASCEND_CUSTOM_PATH})
        if (DEFINED ENV{ALL_IN_ONE_ENABLE})
            set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
            set(ASCEND_COMPILER_DIR ${ASCEND_DIR}/compiler/lib64)
            find_module(slog libalog.so ${ASCEND_COMPILER_DIR})
            find_module(static_mmpa libmmpa.a ${ASCEND_COMPILER_DIR})
        else ()
            set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
            set(ASCEND_ATC_DIR ${ASCEND_DIR}/atc/lib64)
            find_module(slog libalog.so ${ASCEND_ATC_DIR})
            find_module(static_mmpa libmmpa.a ${ASCEND_ATC_DIR})
        endif ()
    endif ()
endif()

if (NOT ENABLE_MS_TESTCASES)
    target_link_libraries(graph_static PRIVATE
        ascend_protobuf_static
    )

    target_link_libraries(register_static PRIVATE
        ascend_protobuf_static
    )
endif()
