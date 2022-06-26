# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# set package information
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_GENERATOR "External")
set(CPACK_CMAKE_GENERATOR "Ninja")
set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${CMAKE_SOURCE_DIR}/build/package/mindspore)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${CMAKE_SOURCE_DIR}/build/package/mindspore)
set(CPACK_PACK_ROOT_DIR ${BUILD_PATH}/package/)
set(CPACK_CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(CPACK_PYTHON_EXE ${Python3_EXECUTABLE})
set(CPACK_PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

if(ENABLE_GPU)
  set(CPACK_MS_BACKEND "ms")
  set(CPACK_MS_TARGET "gpu or cpu")
  if(BUILD_DEV_MODE)
    # providing cuda11 version of dev package only
    set(CPACK_MS_PACKAGE_NAME "mindspore_cuda11_dev")
  else()
    set(CPACK_MS_PACKAGE_NAME "mindspore_gpu")
  endif()
elseif(ENABLE_CPU)
  set(CPACK_MS_BACKEND "ms")
  set(CPACK_MS_TARGET "cpu")
  if(BUILD_DEV_MODE)
    set(CPACK_MS_PACKAGE_NAME "mindspore_dev")
  else()
    set(CPACK_MS_PACKAGE_NAME "mindspore")
  endif()
else()
  set(CPACK_MS_BACKEND "debug")
  set(CPACK_MS_TARGET "ascend or gpu or cpu")
  if(BUILD_DEV_MODE)
    set(CPACK_MS_PACKAGE_NAME "mindspore_dev")
  else()
    set(CPACK_MS_PACKAGE_NAME "mindspore")
  endif()
endif()
include(CPack)

# set install path
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_PY_DIR ".")
set(INSTALL_BASE_DIR ".")
set(INSTALL_BIN_DIR "bin")
set(INSTALL_CFG_DIR "config")

set(INSTALL_LIB_DIR ".")
set(onednn_LIBPATH ${onednn_LIBPATH}/../bin/)
set(glog_LIBPATH ${glog_LIBPATH}/../bin/)
set(opencv_LIBPATH ${opencv_LIBPATH}/../bin/)
set(jpeg_turbo_LIBPATH ${jpeg_turbo_LIBPATH}/../bin/)
set(sqlite_LIBPATH ${sqlite_LIBPATH}/../bin/)
set(tinyxml2_LIBPATH ${tinyxml2_LIBPATH}/../bin/)

message("offline debugger does not support windows system temporarily")

# set package files
install(
  TARGETS _c_expression
  DESTINATION ${INSTALL_BASE_DIR}
  COMPONENT mindspore
)

install(
  TARGETS mindspore_shared_lib
  DESTINATION ${INSTALL_LIB_DIR}
  COMPONENT mindspore
)

install(
  TARGETS mindspore_core mindspore_common mindspore_backend
  DESTINATION ${INSTALL_LIB_DIR}
  COMPONENT mindspore
)

if(USE_GLOG)
  file(GLOB_RECURSE GLOG_LIB_LIST ${glog_LIBPATH}/libmindspore_glog.dll)
  install(
    FILES ${GLOG_LIB_LIST}
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT mindspore
  )
endif()

if(ENABLE_MINDDATA)
  message("icu4c does not support windows system temporarily")
  install(
    TARGETS _c_dataengine _c_mindrecord
    DESTINATION ${INSTALL_BASE_DIR}
    COMPONENT mindspore
  )

  file(GLOB_RECURSE OPENCV_LIB_LIST
    ${opencv_LIBPATH}/libopencv_core*
    ${opencv_LIBPATH}/libopencv_imgcodecs*
    ${opencv_LIBPATH}/libopencv_imgproc*
    )
  install(
    FILES ${OPENCV_LIB_LIST}
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT mindspore
  )
  file(GLOB_RECURSE TINYXML2_LIB_LIST ${tinyxml2_LIBPATH}/libtinyxml2.dll)
  install(
    FILES ${TINYXML2_LIB_LIST}
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT mindspore
  )
endif()

if(ENABLE_CPU)
  file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/dnnl.dll)
  install(
    FILES ${DNNL_LIB_LIST}
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT mindspore
  )
  install(
    TARGETS nnacl
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT mindspore
  )
endif()

if(ENABLE_MPI)
  if(ENABLE_GPU)
    install(
      TARGETS _ms_mpi
      DESTINATION ${INSTALL_BASE_DIR}
      COMPONENT mindspore
    )
  endif()
  if(ENABLE_CPU)
    install(
      TARGETS mpi_adapter
      DESTINATION ${INSTALL_LIB_DIR}
      COMPONENT mindspore
    )
    install(
      TARGETS mpi_collective
      DESTINATION ${INSTALL_LIB_DIR}
      COMPONENT mindspore
    )
  endif()
endif()

if(ENABLE_GPU)
  if(ENABLE_MPI)
    install(
      TARGETS gpu_collective
      DESTINATION ${INSTALL_LIB_DIR}
      COMPONENT mindspore
    )
    install(
      TARGETS nvidia_collective
      DESTINATION ${INSTALL_LIB_DIR}
      COMPONENT mindspore
    )
  endif()
  install(
    TARGETS gpu_queue
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT mindspore
  )
endif()

get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} PATH)
file(GLOB CXX_LIB_LIST ${CXX_DIR}/*.dll)

string(REPLACE "\\" "/" SystemRoot $ENV{SystemRoot})
file(GLOB VC_LIB_LIST ${SystemRoot}/System32/msvcp140.dll ${SystemRoot}/System32/vcomp140.dll)

file(GLOB JPEG_LIB_LIST ${jpeg_turbo_LIBPATH}/*.dll)
file(GLOB SQLITE_LIB_LIST ${sqlite_LIBPATH}/*.dll)
install(
  FILES ${CXX_LIB_LIST} ${JPEG_LIB_LIST} ${SQLITE_LIB_LIST} ${VC_LIB_LIST}
  DESTINATION ${INSTALL_LIB_DIR}
  COMPONENT mindspore
)

# set python files
file(GLOB MS_PY_LIST ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/*.py)
install(
  FILES ${MS_PY_LIST}
  DESTINATION ${INSTALL_PY_DIR}
  COMPONENT mindspore
)

install(
  DIRECTORY
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/nn
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/_extends
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/parallel
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/mindrecord
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/numpy
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/train
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/boost
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/common
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/communication
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/profiler
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/compression
  ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/run_check
  DESTINATION ${INSTALL_PY_DIR}
  COMPONENT mindspore
)

if(EXISTS ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/dataset)
  install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/dataset
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindspore
  )
endif()

## Public header files
install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/include
  DESTINATION ${INSTALL_BASE_DIR}
  COMPONENT mindspore
)

## Public header files for minddata
install(
  FILES ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/config.h
  ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/constants.h
  ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/execute.h
  ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/text.h
  ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/transforms.h
  ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision.h
  ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_lite.h
  ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_ascend.h
  DESTINATION ${INSTALL_BASE_DIR}/include/dataset
  COMPONENT mindspore
)

## config files
install(
  FILES ${CMAKE_SOURCE_DIR}/config/op_info.config
  DESTINATION ${INSTALL_CFG_DIR}
  COMPONENT mindspore
)
