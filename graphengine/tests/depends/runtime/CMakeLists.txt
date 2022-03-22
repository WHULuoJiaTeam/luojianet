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

#cmake_minimum_required(VERSION 2.8)

project(runtime_stub)

file(GLOB_RECURSE SRCS RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "src/runtime_stub.cc"
        )

include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_CODE_DIR}/inc/framework)

add_library(runtime_stub SHARED ${SRCS})

target_compile_options(runtime_stub PRIVATE
    -g
)

target_link_libraries(runtime_stub PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
)

target_include_directories(runtime_stub INTERFACE ${CMAKE_CURRENT_LIST_DIR}/src)
