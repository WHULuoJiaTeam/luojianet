#!/bin/bash
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

set -e
BASEPATH=$(cd "$(dirname $0)"; pwd)
export CUDA_PATH=""
export BUILD_PATH="${BASEPATH}/build/"
export ENABLE_RS="ON"
export ENABLE_GPU="ON"
export ENABLE_MPI="ON"

#remove default patches for third party library
PATCHES_FOLDER="${BUILD_PATH}/luojianet_ms/_ms_patch"
if [ -d $LIB_FOLDER ]; then
     rm -rf $PATCHES_FOLDER
fi

DEPS_FOLDER="${BUILD_PATH}/luojianet_ms/_deps"
if [ -d $DEPS_FOLDER ]; then
    rm -rf ${BUILD_PATH}/luojianet_ms/.mslib
    rm -rf $DEPS_FOLDER/*-build
    rm -rf $DEPS_FOLDER/*-subbuild/CMake*
    rm -rf $DEPS_FOLDER/*-subbuild/Make*
    rm -rf $DEPS_FOLDER/*-subbuild/cmake*
    rm -rf $DEPS_FOLDER/*-subbuild/*-populate-prefix/src/*-stamp
    rm -rf ${BUILD_PATH}/luojianet_ms/cmake*
    rm -rf ${BUILD_PATH}/luojianet_ms/CMake*
fi
