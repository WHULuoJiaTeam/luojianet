file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if(NOT ENABLE_RS)
    SET(_RS_UTILS_SRC_LIST GDALOpenCV.cc GLCM_utils.cc)
    list(REMOVE_ITEM _CURRENT_SRC_FILES ${_RS_UTILS_SRC_LIST})
endif()
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=luojianet_ms::SubModuleId::SM_MD)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(MINGW OR UNIX)
	set(CMAKE_CXX_FLAGS
		  "${CMAKE_CXX_FLAGS} -Wno-unused-result -Wno-delete-incomplete -Wno-unused-variable")
endif()
set(DATASET_CORE_SRC_FILES
        client.cc
        config_manager.cc
        cv_tensor.cc
        data_type.cc
        device_resource.cc
        device_tensor.cc
        de_tensor.cc
        global_context.cc
        tensor.cc
        tensor_helpers.cc
        tensor_row.cc
        tensor_shape.cc
        GDALOpenCV.cc
        GLCM_utils.cc
        )

if(ENABLE_ACL)
  set(DATASET_CORE_SRC_FILES
          ${DATASET_CORE_SRC_FILES}
          ascend_resource.cc)
endif()

ms_protobuf_generate(EXAMPLE_SRCS EXAMPLE_HDRS example.proto)
ms_protobuf_generate(FEATURE_SRCS FEATURE_HDRS feature.proto)
add_library(core OBJECT ${DATASET_CORE_SRC_FILES} ${EXAMPLE_SRCS} ${FEATURE_SRCS})
add_dependencies(core luojianet_ms::protobuf)

if(ENABLE_PYTHON)
  target_include_directories(core PRIVATE ${pybind11_INCLUDE_DIRS})
endif()

if(MSLITE_ENABLE_CLOUD_MIND_DATA)
  add_dependencies(core fbs_src)
endif()
