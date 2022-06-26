include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/core/mindrt/src)

file(GLOB_RECURSE GRAPH_SCHEDULER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if(NOT ENABLE_CPU OR WIN32 OR APPLE)
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "rpc_node_scheduler.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/rpc/recv_actor.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/rpc/rpc_actor.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/rpc/send_actor.cc")
endif()

set_property(SOURCE ${GRAPH_SCHEDULER_SRC_LIST}
  PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=luojianet_ms::SubModuleId::SM_RUNTIME_FRAMEWORK)

add_library(_luojianet_ms_runtime_graph_scheduler_obj OBJECT ${GRAPH_SCHEDULER_SRC_LIST})
