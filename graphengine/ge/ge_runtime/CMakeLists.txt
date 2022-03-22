############ libge_runtime.so ############
set(GE_SRC_LIST
    "model_runner.cc"
    "runtime_model.cc"
    "output.cc"
    "task/aicpu_task.cc"
    "task/cce_task.cc"
    "task/tbe_task.cc"
    "task/event_record_task.cc"
    "task/event_wait_task.cc"
    "task/stream_active_task.cc"
    "task/stream_switch_task.cc"
    "task/hccl_task.cc"
    "task/memcpy_async_task.cc"
    "task/profiler_task.cc"
    "task/label_goto_task.cc"
    "task/label_set_task.cc"
    "task/label_switch_task.cc"
    "task/label_manager.cc"
)

add_library(ge_runtime SHARED ${GE_SRC_LIST})

target_compile_options(ge_runtime PRIVATE
    -Werror
    -O2
    -Wno-deprecated-declarations
    -fno-common
)

target_compile_definitions(ge_runtime PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    LOG_CPP
)

target_include_directories(ge_runtime PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
)

target_link_options(ge_runtime PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_runtime PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    slog
    runtime
    c_sec
    graph
    -Wl,--as-needed
    -lrt
    -ldl
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS ge_runtime OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
