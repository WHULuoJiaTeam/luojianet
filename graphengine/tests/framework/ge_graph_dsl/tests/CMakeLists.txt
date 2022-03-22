# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cc" "*.CC" "*.cpp" "*.CPP")

add_executable(ge_graph_dsl_test ${SOURCES})

target_include_directories(ge_graph_dsl_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(ge_graph_dsl_test PRIVATE
    -g
)
set_target_properties(ge_graph_dsl_test PROPERTIES CXX_STANDARD 17)

target_link_libraries(ge_graph_dsl_test PUBLIC gtest ge_graph_dsl)

include(CTest)
enable_testing()
add_test(NAME test COMMAND ge_graph_dsl_test)