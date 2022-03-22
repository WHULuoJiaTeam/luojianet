file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=luojianet_ms::SubModuleId::SM_MD)
add_library(engine-tdt OBJECT tdt_plugin.cc tdt_handle.cc)