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
project(slog_stub)

file(GLOB_RECURSE SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "src/*.cc"
)
include_directories(${METADEF_DIR}/third_party/fwkacllib/inc)
add_library(slog_stub SHARED ${SRC_FILES})
target_link_libraries(slog_stub PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)
