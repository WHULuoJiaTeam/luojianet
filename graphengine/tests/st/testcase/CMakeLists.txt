file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cc" "*.CC" "*.cpp" "*.CPP" "*.c++")

add_executable(graph_engine_test ${SOURCES})

target_include_directories(graph_engine_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(graph_engine_test PROPERTIES CXX_STANDARD 17)

target_link_libraries(graph_engine_test PRIVATE gtest ge_graph_dsl ge_with_env)

include(CTest)
enable_testing()
add_test(NAME test COMMAND graph_engine_test)