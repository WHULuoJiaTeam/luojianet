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

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cc" "*.CC" "*.cpp" "*.CPP" "*.c++")

# ---- Target : stub Host engine ----
add_library(ge_with_env STATIC ${SOURCES})

target_include_directories(ge_with_env
        PUBLIC
        include
        )
        
target_include_directories(ge_with_env
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        )

target_compile_definitions(ge_with_env PRIVATE
        google=ascend_private
        FMK_SUPPORT_DUMP
        )

target_compile_options(ge_with_env PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -Werror=format
        )

target_link_libraries(ge_with_env PUBLIC
        $<BUILD_INTERFACE:intf_pub> ge_running_env_inc graphengine -lrt -ldl -lpthread -lgcov
        )

set_target_properties(ge_with_env PROPERTIES CXX_STANDARD 17)
