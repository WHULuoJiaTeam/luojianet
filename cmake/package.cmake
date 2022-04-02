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

if(ENABLE_GE)
    set(CPACK_MS_BACKEND "ge")
    set(CPACK_MS_TARGET "ascend or cpu")
    set(CPACK_MS_PACKAGE_NAME "luojianet_ms")
elseif(ENABLE_GPU)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "gpu or cpu")
    if(BUILD_DEV_MODE)
        # providing cuda11 version of dev package only
        set(CPACK_MS_PACKAGE_NAME "luojianet_ms-cuda11-dev")
    else()
        set(CPACK_MS_PACKAGE_NAME "luojianet_ms-gpu")
    endif()
elseif(ENABLE_D)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "ascend or cpu")
    if(BUILD_DEV_MODE)
        set(CPACK_MS_PACKAGE_NAME "luojianet_ms-ascend-dev")
    else()
        set(CPACK_MS_PACKAGE_NAME "luojianet_ms-ascend")
    endif()
elseif(ENABLE_CPU)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "cpu")
    if(BUILD_DEV_MODE)
        set(CPACK_MS_PACKAGE_NAME "luojianet_ms-dev")
    else()
        set(CPACK_MS_PACKAGE_NAME "luojianet_ms")
    endif()
elseif(ENABLE_ACL)
    set(CPACK_MS_BACKEND "debug")
    set(CPACK_MS_TARGET "ascend or gpu or cpu")
    if(BUILD_DEV_MODE)
        set(CPACK_MS_PACKAGE_NAME "luojianet_ms-ascend-dev")
    else()
        set(CPACK_MS_PACKAGE_NAME "luojianet_ms-ascend")
    endif()
else()
    set(CPACK_MS_BACKEND "debug")
    set(CPACK_MS_TARGET "ascend or gpu or cpu")
    if(BUILD_DEV_MODE)
        set(CPACK_MS_PACKAGE_NAME "luojianet_ms-dev")
    else()
        set(CPACK_MS_PACKAGE_NAME "luojianet_ms")
    endif()
endif()
include(CPack)

if (NOT ${GENERATE_RESULT} EQUAL 0)
  message(FATAL_ERROR "generate.py failed.")
endif ()

# set install path
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_PY_DIR ".")
set(INSTALL_BASE_DIR ".")
set(INSTALL_BIN_DIR "bin")
set(INSTALL_CFG_DIR "config")
set(INSTALL_LIB_DIR "lib")

if(ENABLE_RS)
    set(gdal_ext_LIBPATH ${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/lib/)
    set(gdal_extra_LIBPATH ${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/third_party/lib/)
endif()

# set package files
install(
    TARGETS _c_expression
    DESTINATION ${INSTALL_BASE_DIR}
    COMPONENT luojianet_ms
)

if(ENABLE_DEBUGGER)
    install(
        TARGETS _luojianet_offline_debug
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
    install(FILES ${glog_LIBPATH}/libluojianet_ms_glog.so.0.4.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libluojianet_ms_glog.so.0 COMPONENT luojianet_ms)
endif()

install(FILES ${libevent_LIBPATH}/libevent-2.1.so.7.0.1
  DESTINATION ${INSTALL_LIB_DIR} RENAME libevent-2.1.so.7 COMPONENT luojianet_ms)
install(FILES ${libevent_LIBPATH}/libevent_core-2.1.so.7.0.1
  DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_core-2.1.so.7 COMPONENT luojianet_ms)
install(FILES ${libevent_LIBPATH}/libevent_extra-2.1.so.7.0.1
  DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_extra-2.1.so.7 COMPONENT luojianet_ms)
install(FILES ${libevent_LIBPATH}/libevent_openssl-2.1.so.7.0.1
  DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_openssl-2.1.so.7 COMPONENT luojianet_ms)
install(FILES ${libevent_LIBPATH}/libevent_pthreads-2.1.so.7.0.1
  DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_pthreads-2.1.so.7 COMPONENT luojianet_ms)

if(ENABLE_MINDDATA)
    install(
        TARGETS _c_dataengine _c_mindrecord
        DESTINATION ${INSTALL_BASE_DIR}
        COMPONENT luojianet_ms
    )
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        install(
            TARGETS cache_admin cache_server
            OPTIONAL
            DESTINATION ${INSTALL_BIN_DIR}
            COMPONENT luojianet_ms
        )
    endif()
    if(PYTHON_VERSION MATCHES "3.8" OR PYTHON_VERSION MATCHES "3.7")
      install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.2.0
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_core.so.4.2 COMPONENT luojianet_ms)
      install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.2.0
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgcodecs.so.4.2 COMPONENT luojianet_ms)
      install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.2.0
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgproc.so.4.2 COMPONENT luojianet_ms)
    elseif(PYTHON_VERSION MATCHES "3.9")
      install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_core.so.4.5 COMPONENT luojianet_ms)
      install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgcodecs.so.4.5 COMPONENT luojianet_ms)
      install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgproc.so.4.5 COMPONENT luojianet_ms)
    endif()

    if(ENABLE_RS)
        if(PYTHON_VERSION MATCHES "3.8" OR PYTHON_VERSION MATCHES "3.7")
            install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.2.0
                    DESTINATION ${INSTALL_LIB_DIR}/.. RENAME libopencv_core.so.4.2 COMPONENT luojianet_ms)
            install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.2.0
                    DESTINATION ${INSTALL_LIB_DIR}/.. RENAME libopencv_imgcodecs.so.4.2 COMPONENT luojianet_ms)
            install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.2.0
                    DESTINATION ${INSTALL_LIB_DIR}/.. RENAME libopencv_imgproc.so.4.2 COMPONENT luojianet_ms)
        elseif(PYTHON_VERSION MATCHES "3.9")
            install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.1
                    DESTINATION ${INSTALL_LIB_DIR}/.. RENAME libopencv_core.so.4.5 COMPONENT luojianet_ms)
            install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.1
                    DESTINATION ${INSTALL_LIB_DIR}/.. RENAME libopencv_imgcodecs.so.4.5 COMPONENT luojianet_ms)
            install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.1
                    DESTINATION ${INSTALL_LIB_DIR}/.. RENAME libopencv_imgproc.so.4.5 COMPONENT luojianet_ms)
        endif()
    endif()

    install(FILES ${tinyxml2_LIBPATH}/libtinyxml2.so.8.0.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libtinyxml2.so.8 COMPONENT luojianet_ms)

    install(FILES ${icu4c_LIBPATH}/libicuuc.so.67.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicuuc.so.67 COMPONENT luojianet_ms)
    install(FILES ${icu4c_LIBPATH}/libicudata.so.67.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicudata.so.67 COMPONENT luojianet_ms)
    install(FILES ${icu4c_LIBPATH}/libicui18n.so.67.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicui18n.so.67 COMPONENT luojianet_ms)
endif()

if(ENABLE_CPU)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        install(FILES ${onednn_LIBPATH}/libdnnl.so.2.2
          DESTINATION ${INSTALL_LIB_DIR} RENAME libdnnl.so.2 COMPONENT luojianet_ms)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/libdnnl*${CMAKE_SHARED_LIBRARY_SUFFIX}*)
        install(
            FILES ${DNNL_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
        )
    endif()
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
        install(
          TARGETS mpi_collective
          DESTINATION ${INSTALL_LIB_DIR}
          COMPONENT luojianet_ms
        )
    endif()
    if(ENABLE_D)
        install(
                TARGETS _ascend_mpi
                DESTINATION ${INSTALL_BASE_DIR}
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
        install(
          TARGETS nvidia_collective
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

if(ENABLE_D)
    if(ENABLE_MPI)
        install(
                TARGETS ascend_collective
                DESTINATION ${INSTALL_LIB_DIR}
                COMPONENT luojianet_ms
        )
    endif()
endif()

if(ENABLE_CPU AND NOT WIN32)
    install(
        TARGETS ps_cache
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT luojianet_ms
    )
endif()

if(ENABLE_RS)
    install(
            TARGETS geobject
            DESTINATION ${INSTALL_BASE_DIR}
            COMPONENT luojianet_ms
    )
endif()

if(NOT ENABLE_GE)
    if(ENABLE_D OR ENABLE_ACL)
        if(DEFINED ENV{ASCEND_CUSTOM_PATH})
            set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
        else()
            set(ASCEND_PATH /usr/local/Ascend)
        endif()
        set(ASCEND_DRIVER_PATH ${ASCEND_PATH}/driver/lib64/common)

        if(ENABLE_D)
            install(
              TARGETS hccl_plugin
              DESTINATION ${INSTALL_LIB_DIR}
              COMPONENT luojianet_ms
            )
        endif()
    elseif(ENABLE_TESTCASES)
        install(
            FILES
                ${CMAKE_BINARY_DIR}/graphengine/metadef/graph/libgraph.so
                ${BUILD_PATH}/graphengine/c_sec/lib/libc_sec.so
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
        )
    endif()
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

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} PATH)
    file(GLOB CXX_LIB_LIST ${CXX_DIR}/*.dll)

    string(REPLACE "\\" "/" SystemRoot $ENV{SystemRoot})
    file(GLOB VC_LIB_LIST ${SystemRoot}/System32/msvcp140.dll ${SystemRoot}/System32/vcomp140.dll)

    file(GLOB JPEG_LIB_LIST ${jpeg_turbo_LIBPATH}/*.dll)
    file(GLOB SQLITE_LIB_LIST ${sqlite_LIBPATH}/*.dll)
    install(
        FILES ${CXX_LIB_LIST} ${JPEG_LIB_LIST} ${SQLITE_LIB_LIST} ${VC_LIB_LIST}
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT luojianet_ms
    )
endif()

if(ENABLE_RS)
    file(GLOB GDAL_LIB_LIST "${gdal_ext_LIBPATH}/lib*")
    install(
            FILES ${GDAL_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
    )

    file(GLOB GDAL_EXTRA_LIB_LIST "${gdal_extra_LIBPATH}/lib*")
    install(
            FILES ${GDAL_EXTRA_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT luojianet_ms
    )

    file(GLOB GDAL_LIB_LIST_SO "${gdal_ext_LIBPATH}/lib*")
    install(
            FILES ${GDAL_LIB_LIST_SO}
            DESTINATION ${INSTALL_LIB_DIR}/..
            COMPONENT luojianet_ms
    )

    file(GLOB GDAL_EXTRA_LIB_LIST_SO "${gdal_extra_LIBPATH}/lib*")
    install(
            FILES ${GDAL_EXTRA_LIB_LIST_SO}
            DESTINATION ${INSTALL_LIB_DIR}/..
            COMPONENT luojianet_ms
    )


#    file(GLOB GDAL_LIB_LIST "${gdal_LIBPATH}/libgdal*")
#    install(
#            FILES ${GDAL_LIB_LIST}
#            DESTINATION ${INSTALL_LIB_DIR}
#            COMPONENT luojianet_ms
#    )

    # copy over python files
    file(GLOB_RECURSE MDP_PY_FILES ${CMAKE_SOURCE_DIR}/dataset_plugin/*.py)

    install(
            FILES ${MDP_PY_FILES} ${CMAKE_SOURCE_DIR}/setup.py
            DESTINATION ${INSTALL_PY_DIR}
            COMPONENT luojianet_ms
    )
endif()


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

if(ENABLE_AKG AND CMAKE_SYSTEM_NAME MATCHES "Linux")
    set (AKG_PATH ${BUILD_PATH}/luojianet_ms/akg)
    file(REMOVE_RECURSE ${AKG_PATH}/_akg)
    file(MAKE_DIRECTORY ${AKG_PATH}/_akg)
    file(TOUCH ${AKG_PATH}/_akg/__init__.py)
    install(DIRECTORY "${AKG_PATH}/akg" DESTINATION "${AKG_PATH}/_akg")
    install(
        DIRECTORY
            ${AKG_PATH}/_akg
        DESTINATION ${INSTALL_PY_DIR}/
        COMPONENT luojianet_ms
    )
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/dataset)
    install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/dataset
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT luojianet_ms
    )
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message("offline debugger does not support windows system temporarily")
else()
    if(EXISTS ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/offline_debug)
        install(
            DIRECTORY ${CMAKE_SOURCE_DIR}/luojianet_ms/python/luojianet_ms/offline_debug
            DESTINATION ${INSTALL_PY_DIR}
            COMPONENT luojianet_ms
        )
    endif()
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

install(
    FILES
        ${CMAKE_SOURCE_DIR}/luojianet_ms/core/mindapi/base/format.h
        ${CMAKE_SOURCE_DIR}/luojianet_ms/core/mindapi/base/type_id.h
        ${CMAKE_SOURCE_DIR}/luojianet_ms/core/mindapi/base/types.h
    DESTINATION ${INSTALL_BASE_DIR}/include/mindapi/base
    COMPONENT luojianet_ms)

## config files
install(
    FILES ${CMAKE_SOURCE_DIR}/config/op_info.config
    DESTINATION ${INSTALL_CFG_DIR}
    COMPONENT luojianet_ms
)
