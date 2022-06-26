file(GLOB_RECURSE HARDWARE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "device_context_manager.cc")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()

set_property(SOURCE ${HARDWARE_SRC_LIST} PROPERTY SUBMODULE_ID=luojianet_ms::SubModuleId::SM_DEVICE)
add_library(_luojianet_ms_runtime_hardware_obj OBJECT ${HARDWARE_SRC_LIST})
