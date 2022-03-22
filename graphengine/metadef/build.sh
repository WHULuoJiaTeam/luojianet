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
OUTPUT_PATH="${BASEPATH}/output"
export BUILD_PATH="${BASEPATH}/build/"

# print usage message
usage()
{
  echo "Usage:"
  echo "sh build.sh [-j[n]] [-h] [-v] [-s] [-t] [-u] [-c] [-S on|off]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -u Only compile ut, not execute"
  echo "    -s Build st"
  echo "    -j[n] Set the number of threads used for building Metadef, default is 8"
  echo "    -t Build and execute ut"
  echo "    -c Build ut with coverage tag"
  echo "    -v Display build command"
  echo "    -S Enable enable download cmake compile dependency from gitee , default off"
  echo "to be continued ..."
}

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

# parse and set options
checkopts()
{
  VERBOSE=""
  THREAD_NUM=8
  # ENABLE_METADEF_UT_ONLY_COMPILE="off"
  ENABLE_METADEF_UT="off"
  ENABLE_METADEF_ST="off"
  ENABLE_METADEF_COV="off"
  GE_ONLY="on"
  ENABLE_GITEE="off"
  # Process the options
  while getopts 'ustchj:vS:' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      u)
        ENABLE_METADEF_UT="on"
        GE_ONLY="off"
        ;;
      s)
        ENABLE_METADEF_ST="on"
        ;;
      t)
        ENABLE_METADEF_UT="on"
        GE_ONLY="off"
        ;;
      c)
        ENABLE_METADEF_COV="on"
        GE_ONLY="off"
        ;;
      h)
        usage
        exit 0
        ;;
      j)
        THREAD_NUM=$OPTARG
        ;;
      v)
        VERBOSE="VERBOSE=1"
        ;;
      S)
        check_on_off $OPTARG S
        ENABLE_GITEE="$OPTARG"
        echo "enable download from gitee"
        ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
    esac
  done
}
checkopts "$@"

mk_dir() {
    local create_dir="$1"  # the target to make

    mkdir -pv "${create_dir}"
    echo "created ${create_dir}"
}

# Meatdef build start
echo "---------------- Metadef build start ----------------"

# create build path
build_metadef()
{
  echo "create build directory and build Metadef";
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"
  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DGE_ONLY=$GE_ONLY"

  if [[ "X$ENABLE_METADEF_COV" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_METADEF_COV=ON"
  fi

  if [[ "X$ENABLE_METADEF_UT" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_METADEF_UT=ON"
  fi


  if [[ "X$ENABLE_METADEF_ST" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_METADEF_ST=ON"
  fi

  if [[ "X$ENABLE_GITEE" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GITEE=ON"
  fi
  
  CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_OPEN_SRC=True -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH}"
  echo "${CMAKE_ARGS}"
  cmake ${CMAKE_ARGS} ..
  if [ 0 -ne $? ]
  then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    return 1
  fi
  
  if [ "X$ENABLE_METADEF_UT" = "Xon" ]; then
    make ut_graph ut_register -j${THREAD_NUM}
  else
    make ${VERBOSE} -j${THREAD_NUM} && make install
  fi
  if [ 0 -ne $? ]
  then
    echo "execute command: make ${VERBOSE} -j${THREAD_NUM} && make install failed."
    return 1
  fi
  echo "Metadef build success!"
}

g++ -v
mk_dir ${OUTPUT_PATH}
build_metadef || { echo "Metadef build failed."; return; }
echo "---------------- Metadef build finished ----------------"
rm -f ${OUTPUT_PATH}/libgmock*.so
rm -f ${OUTPUT_PATH}/libgtest*.so
rm -f ${OUTPUT_PATH}/lib*_stub.so

chmod -R 750 ${OUTPUT_PATH}
find ${OUTPUT_PATH} -name "*.so*" -print0 | xargs -0 chmod 500

echo "---------------- Metadef output generated ----------------"

if [[ "X$ENABLE_METADEF_UT" = "Xon" || "X$ENABLE_METADEF_COV" = "Xon" ]]; then
    cp ${BUILD_PATH}/tests/ut/graph/ut_graph ${OUTPUT_PATH}
    cp ${BUILD_PATH}/tests/ut/register/ut_register ${OUTPUT_PATH}

    RUN_TEST_CASE=${OUTPUT_PATH}/ut_graph && ${RUN_TEST_CASE} &&
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_register && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi
    echo "Generating coverage statistics, please wait..."
    cd ${BASEPATH}
    rm -rf ${BASEPATH}/cov
    mkdir ${BASEPATH}/cov
    lcov -c -d build/graph/CMakeFiles/graph_static.dir -d build/register/CMakeFiles/register_static.dir/ -o cov/tmp.info
    lcov -r cov/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/tests/*' '/usr/*' '*/ops/*' '*/register/graph_optimizer/*' -o cov/coverage.info
    cd ${BASEPATH}/cov
    genhtml coverage.info
fi

# generate output package in tar form, including ut/st libraries/executables
generate_package()
{
  cd "${BASEPATH}"

  METADEF_LIB_PATH="lib"
  ACL_PATH="acllib/lib64"
  FWK_PATH="fwkacllib/lib64"
  ATC_PATH="atc/lib64"

  COMMON_LIB=("libgraph.so" "libregister.so" "liberror_manager.so")

  rm -rf ${OUTPUT_PATH:?}/${FWK_PATH}/
  rm -rf ${OUTPUT_PATH:?}/${ACL_PATH}/
  rm -rf ${OUTPUT_PATH:?}/${ATC_PATH}/
  
  mk_dir "${OUTPUT_PATH}/${FWK_PATH}"
  mk_dir "${OUTPUT_PATH}/${ATC_PATH}"
  mk_dir "${OUTPUT_PATH}/${ACL_PATH}"

  find output/ -name metadef_lib.tar -exec rm {} \;

  cd "${OUTPUT_PATH}"

  for lib in "${COMMON_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${METADEF_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${FWK_PATH} \;
    find ${OUTPUT_PATH}/${METADEF_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${ATC_PATH} \;
  done

  find ${OUTPUT_PATH}/${METADEF_LIB_PATH} -maxdepth 1 -name "libc_sec.so" -exec cp -f {} ${OUTPUT_PATH}/${ATC_PATH} \;

  tar -cf metadef_lib.tar fwkacllib atc
}

# generate output package in tar form, including ut/st libraries/executables for cann
generate_package_for_cann()
{
  cd "${BASEPATH}"

  METADEF_LIB_PATH="lib"
  COMPILER_PATH="compiler/lib64"
  COMMON_LIB=("libgraph.so" "libregister.so" "liberror_manager.so")

  rm -rf ${OUTPUT_PATH:?}/${COMPILER_PATH}/

  mk_dir "${OUTPUT_PATH}/${COMPILER_PATH}"

  find output/ -name metadef_lib.tar -exec rm {} \;

  cd "${OUTPUT_PATH}"

  for lib in "${COMMON_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${METADEF_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH} \;
  done

  find ${OUTPUT_PATH}/${METADEF_LIB_PATH} -maxdepth 1 -name "libc_sec.so" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH} \;

  tar -cf metadef_lib.tar compiler
}

if [[ "X$ENABLE_METADEF_UT" = "Xoff" ]]; then
  if [[ "X$ALL_IN_ONE_ENABLE" = "X1" ]]; then
    generate_package_for_cann
  else
    generate_package
  fi
fi
echo "---------------- Metadef package archive generated ----------------"
