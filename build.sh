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

source ./scripts/build/usage.sh
source ./scripts/build/default_options.sh
source ./scripts/build/option_proc_debug.sh
source ./scripts/build/option_proc_luojianet.sh
source ./scripts/build/option_proc_lite.sh
source ./scripts/build/process_options.sh
source ./scripts/build/parse_device.sh
source ./scripts/build/build_luojianet.sh

#remove default patches for third party library
PATCHES_FOLDER="${BUILD_PATH}/luojianet_ms/_ms_patch"
if [ ! -d $LIB_FOLDER ]; then
     rm -rf $PATCHES_FOLDER
fi

DEPS_FOLDER="${BUILD_PATH}/luojianet_ms/_deps"
if [ ! -d $DEPS_FOLDER ]; then
    rm -rf ${BUILD_PATH}/luojianet_ms/.mslib
    rm -rf $DEPS_FOLDER
    rm -rf $DEPS_FOLDER/*-src
    rm -rf $DEPS_FOLDER/*-build
    rm -rf $DEPS_FOLDER/*-subbuild/CMake*
    rm -rf $DEPS_FOLDER/*-subbuild/Make*
    rm -rf $DEPS_FOLDER/*-subbuild/cmake*
    rm -rf $DEPS_FOLDER/*-subbuild/*-populate-prefix/src/*-stamp
    rm -rf ${BUILD_PATH}/luojianet_ms/cmake*
    rm -rf ${BUILD_PATH}/luojianet_ms/CMake*
fi

# check value of input is 'on' or 'off'
# usage: check_on_off arg_value arg_name
check_on_off()
{
  if [[ "X$1" != "Xon" && "X$1" != "Xoff" ]]; then
    echo "Invalid value $1 for option -$2"
    usage
    exit 1
  fi
}

update_submodule()
{
  git submodule update --init graphengine
  cd "${BASEPATH}/graphengine"
  git submodule update --init metadef
  cd "${BASEPATH}"
  if [[ "X$ENABLE_AKG" = "Xon" ]]; then
    if [[ "X$ENABLE_D" == "Xon" ]]; then
      git submodule update --init akg
    else
      GIT_LFS_SKIP_SMUDGE=1 git submodule update --init akg
    fi
  fi
}

build_exit()
{
    echo "$@" >&2
    stty echo
    exit 1
}

make_clean()
{
  echo "enable make clean"
  cd "${BUILD_PATH}/luojianet_ms"
  cmake --build . --target clean
}

echo "---------------- LuoJiaNet: build start ----------------"
init_default_options
process_options "$@"
parse_device

if [[ "X$COMPILE_LITE" = "Xon" ]]; then
  export COMPILE_MINDDATA_LITE
  export ENABLE_VERBOSE
  export LITE_PLATFORM
  export LITE_ENABLE_AAR
  source luojianet_ms/lite/build_lite.sh
else
  mkdir -pv "${BUILD_PATH}/package/luojianet_ms/lib"
  # update_submodule

  build_luojianet

  if [[ "X$ENABLE_MAKE_CLEAN" = "Xon" ]]; then
    make_clean
  fi
  if [[ "X$ENABLE_ACL" == "Xon" ]] && [[ "X$ENABLE_D" == "Xoff" ]]; then
      echo "acl mode, skipping deploy phase"
      rm -rf ${BASEPATH}/output/_CPack_Packages/
    else
      cp -rf ${BUILD_PATH}/package/luojianet_ms/lib ${BASEPATH}/luojianet_ms/python/luojianet_ms
      cp -rf ${BUILD_PATH}/package/luojianet_ms/*.so ${BASEPATH}/luojianet_ms/python/luojianet_ms
  fi
fi
echo "---------------- LuoJiaNet: build end   ----------------"
