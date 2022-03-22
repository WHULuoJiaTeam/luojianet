target_link_libraries(graph_static PRIVATE
    ascend_protobuf
)

target_link_libraries(register_static PRIVATE
    ascend_protobuf)

target_compile_options(graph_static PRIVATE
    -g --coverage -fprofile-arcs -fPIC -O0 -ftest-coverage)


target_compile_options(register_static PRIVATE
    -g --coverage -fprofile-arcs -fPIC -O0 -ftest-coverage)