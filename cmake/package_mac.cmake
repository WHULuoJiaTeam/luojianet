# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# set package information
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_GENERATOR "External")
set(CPACK_CMAKE_GENERATOR "Ninja")
set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${BUILD_PATH}/package/luojianet_ms)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${BUILD_PATH}/package/luojianet_ms)
set(CPACK_PACK_ROOT_DIR ${BUILD_PATH}/package/)
set(CPACK_CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(CPACK_ENABLE_SYM_FILE ${ENABLE_SYM_FILE})
set(CPACK_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
set(CPACK_PYTHON_EXE ${Python3_EXECUTABLE})
set(CPACK_PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})
set(CPACK_OSX_DEPLOYMENT_TARGET ${CMAKE_OSX_DEPLOYMENT_TARGET})


if(ENABLE_GPU)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "gpu or cpu")
    set(CPACK_MS_PACKAGE_NAME "luojianet_ms-gpu")
elseif(ENABLE_CPU)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "cpu")
    set(CPACK_MS_PACKAGE_NAME "luojianet_ms")
else()
    set(CPACK_MS_BACKEND "debug")
    set(CPACK_MS_TARGET "ascend or gpu or cpu")
    set(CPACK_MS_PACKAGE_NAME "luojianet_ms")
endif()
include(CPack)

# set install path
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_PY_DIR ".")
set(INSTALL_BASE_DIR ".")
set(INSTALL_BIN_DIR "bin")
set(INSTALL_CFG_DIR "config")
set(INSTALL_LIB_DIR "lib")
# set package files
install(
    TARGETS _c_expression
    DESTINATION ${INSTALL_BASE_DIR}
    COMPONENT luojianet_ms
)

if(ENABLE_DEBUGGER)
    install(
        TARGETS _luojianet_ms_offline_debug
        DESTINATION ${INSTALL_BASE_DIR}
        COMPONENT luojianet_ms
    )
endif()

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

install(FILES ${libevent_LIBPATH}/libevent-2.1.7.dylib
  DESTINATION ${INSTALL_LIB_DIR} COMPONENT luojianet_ms)
install(FILES ${libevent_LIBPATH}/libevent_core-2.1.7.dylib
  DESTINATION ${INSTALL_LIB_DIR} COMPONENT luojianet_ms)
install(FILES ${libevent_LIBPATH}/libevent_extra-2.1.7.dylib
  DESTINATION ${INSTALL_LIB_DIR} COMPONENT luojianet_ms)
install(FILES ${libevent_LIBPATH}/libevent_openssl-2.1.7.dylib
  DESTINATION ${INSTALL_LIB_DIR} COMPONENT luojianet_ms)
install(FILES ${libevent_LIBPATH}/libevent_pthreads-2.1.7.dylib
  DESTINATION ${INSTALL_LIB_DIR} COMPONENT luojianet_ms)

if(ENABLE_CPU AND NOT WIN32)
    install(
        TARGETS ps_cache
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT luojianet_ms
    )
endif()

if(MS_BUILD_GRPC)
    file(GLOB_RECURSE GRPC_LIB_LIST
        ${grpc_LIBPATH}/libluojianet_ms*
    )
    install(
        FILES ${GRPC_LIB_LIST}
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT luojianet_ms
    )
endif()

if(ENABLE_MINDDATA)
    install(
        TARGETS _c_dataengine _c_mindrecord
        DESTINATION ${INSTALL_BASE_DIR}
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
    file(GLOB_RECURSE JPEG_LIB_LIST
        ${jpeg_turbo_LIBPATH}/*.dylib
    )
    install(
        FILES ${JPEG_LIB_LIST}
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

if(ENABLE_CPU)
    file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/libdnnl*)
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

if(ENABLE_MPI)
    if(ENABLE_GPU)
        install(
            TARGETS _ms_mpi
            DESTINATION ${INSTALL_BASE_DIR}
            COMPONENT luojianet_ms
        )
  endif()
    if(ENABLE_CPU)
        install(
            TARGETS mpi_adapter
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
        )
    endif()
endif()

if(ENABLE_GPU)
    if(ENABLE_MPI)
        install(
            TARGETS gpu_collective
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
        )
    endif()
    install(
        TARGETS gpu_queue
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT luojianet_ms
    )
endif()

get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} PATH)
file(GLOB CXX_LIB_LIST ${CXX_DIR}/*.dylib)

file(GLOB JPEG_LIB_LIST ${jpeg_turbo_LIBPATH}/*.dylib)
file(GLOB SQLITE_LIB_LIST ${sqlite_LIBPATH}/*.dylib)
install(
    FILES ${CXX_LIB_LIST} ${SQLITE_LIB_LIST}
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT luojianet_ms
)

# set python files
file(GLOB MS_PY_LIST ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/*.py)
install(
    FILES ${MS_PY_LIST}
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT luojianet_ms
)

install(
    DIRECTORY
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/nn
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/_extends
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/parallel
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/mindrecord
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/numpy
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/scipy
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/train
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/boost
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/common
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/ops
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/communication
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/profiler
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/compression
    ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/run_check
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT luojianet_ms
)

if(EXISTS ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/dataset)
    install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/dataset
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT luojianet_ms
    )
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/offline_debug)
    install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/offline_debug
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT luojianet_ms
    )
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
