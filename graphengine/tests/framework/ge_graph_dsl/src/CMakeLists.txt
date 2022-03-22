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

add_library(ge_graph_dsl STATIC ${SOURCES})

target_include_directories(ge_graph_dsl	
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ge_graph_dsl PUBLIC
    $<BUILD_INTERFACE:intf_pub>)

set_target_properties(ge_graph_dsl PROPERTIES CXX_STANDARD 17)
target_link_libraries(ge_graph_dsl PUBLIC ge_graph_dsl_inc easy_graph metadef_graph)
