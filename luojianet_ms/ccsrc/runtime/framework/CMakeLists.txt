include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/core/mindrt/src)

file(GLOB_RECURSE FRAMEWORK_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

set_property(SOURCE ${FRAMEWORK_SRC_LIST}
  PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=luojianet_ms::SubModuleId::SM_RUNTIME_FRAMEWORK)
add_library(_luojianet_runtime_framework_obj OBJECT ${FRAMEWORK_SRC_LIST})