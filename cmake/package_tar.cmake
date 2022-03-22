# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# prepare output directory
file(REMOVE_RECURSE ${CMAKE_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

# cpack variables
string(TOLOWER linux_${CMAKE_HOST_SYSTEM_PROCESSOR} PLATFORM_NAME)
set(CPACK_PACKAGE_FILE_NAME luojianet_ms_ascend-${VERSION_NUMBER}-${PLATFORM_NAME})
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_BASE_DIR ".")
set(INSTALL_BIN_DIR "bin")
set(INSTALL_CFG_DIR "config")
set(INSTALL_LIB_DIR "lib")

# set package files
install(
        TARGETS luojianet_shared_lib
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT luojianet_ms
)

install(
        TARGETS luojianet_gvar
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT luojianet_ms
)

if(USE_GLOG)
    file(GLOB_RECURSE GLOG_LIB_LIST ${glog_LIBPATH}/libluojianet_ms_glog*)
    install(
            FILES ${GLOG_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
    )
endif()

file(GLOB_RECURSE LIBEVENT_LIB_LIST
        ${libevent_LIBPATH}/libevent*${CMAKE_SHARED_LIBRARY_SUFFIX}*
        ${libevent_LIBPATH}/libevent_pthreads*${CMAKE_SHARED_LIBRARY_SUFFIX}*
        )

install(
        FILES ${LIBEVENT_LIB_LIST}
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT luojianet_ms
)

if(ENABLE_MINDDATA)
    install(
            TARGETS _c_dataengine _c_mindrecord
            DESTINATION ${INSTALL_BASE_DIR}
            COMPONENT luojianet_ms
    )
    install(
            TARGETS cache_admin cache_server
            OPTIONAL
            DESTINATION ${INSTALL_BIN_DIR}
            COMPONENT luojianet_ms
    )
    file(GLOB_RECURSE OPENCV_LIB_LIST
            ${opencv_LIBPATH}/libopencv_core*
            ${opencv_LIBPATH}/libopencv_imgcodecs*
            ${opencv_LIBPATH}/libopencv_imgproc*
            )
    install(
            FILES ${OPENCV_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
    )
    file(GLOB_RECURSE TINYXML2_LIB_LIST ${tinyxml2_LIBPATH}/libtinyxml2*)
    install(
            FILES ${TINYXML2_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
    )
    file(GLOB_RECURSE ICU4C_LIB_LIST
            ${icu4c_LIBPATH}/libicuuc*
            ${icu4c_LIBPATH}/libicudata*
            ${icu4c_LIBPATH}/libicui18n*
            )
    install(
            FILES ${ICU4C_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
    )
endif()

# CPU mode
if(ENABLE_CPU AND NOT WIN32)
    install(
            TARGETS ps_cache
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
    )
endif()

if(ENABLE_CPU)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/libdnnl${CMAKE_SHARED_LIBRARY_SUFFIX}*)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/libdnnl*${CMAKE_SHARED_LIBRARY_SUFFIX}*)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
        file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/dnnl.dll)
    endif()
    install(
            FILES ${DNNL_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
    )
    install(
            TARGETS nnacl
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
    )
endif()

if(MS_BUILD_GRPC)
    install(FILES ${grpc_LIBPATH}/libluojianet_ms_grpc++.so.1.36.1
            DESTINATION ${INSTALL_LIB_DIR} RENAME libluojianet_ms_grpc++.so.1 COMPONENT luojianet_ms)
    install(FILES ${grpc_LIBPATH}/libluojianet_ms_grpc.so.15.0.0
            DESTINATION ${INSTALL_LIB_DIR} RENAME libluojianet_ms_grpc.so.15 COMPONENT luojianet_ms)
    install(FILES ${grpc_LIBPATH}/libluojianet_ms_gpr.so.15.0.0
            DESTINATION ${INSTALL_LIB_DIR} RENAME libluojianet_ms_gpr.so.15 COMPONENT luojianet_ms)
    install(FILES ${grpc_LIBPATH}/libluojianet_ms_upb.so.15.0.0
            DESTINATION ${INSTALL_LIB_DIR} RENAME libluojianet_ms_upb.so.15 COMPONENT luojianet_ms)
    install(FILES ${grpc_LIBPATH}/libluojianet_ms_address_sorting.so.15.0.0
            DESTINATION ${INSTALL_LIB_DIR} RENAME libluojianet_ms_address_sorting.so.15 COMPONENT luojianet_ms)
endif()

## Public header files
install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/include
        DESTINATION ${INSTALL_BASE_DIR}
        COMPONENT luojianet_ms
)

## Public header files for minddata
install(
        FILES ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/include/dataset/config.h
        ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/include/dataset/constants.h
        ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/include/dataset/execute.h
        ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/include/dataset/text.h
        ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/include/dataset/transforms.h
        ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/include/dataset/vision.h
        ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/include/dataset/vision_lite.h
        ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/include/dataset/vision_ascend.h
        DESTINATION ${INSTALL_BASE_DIR}/include/dataset
        COMPONENT luojianet_ms
)

## config files
install(
        FILES ${CMAKE_SOURCE_DIR}/config/op_info.config
        DESTINATION ${INSTALL_CFG_DIR}
        COMPONENT luojianet_ms
)

include(CPack)
