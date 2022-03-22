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
  echo "sh build.sh [-j[n]] [-h] [-v] [-s] [-t] [-u] [-c] [-S on|off] [-M]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -u Only compile ut, not execute"
  echo "    -s Build st"
  echo "    -j[n] Set the number of threads used for building GraphEngine, default is 8"
  echo "    -t Build and execute ut"
  echo "    -c Build ut with coverage tag"
  echo "    -p Build inference or train"
  echo "    -v Display build command"
  echo "    -S Enable enable download cmake compile dependency from gitee , default off"
  echo "    -M build LuoJiaNet mode"
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
  # ENABLE_GE_UT_ONLY_COMPILE="off"
  ENABLE_GE_UT="off"
  ENABLE_GE_ST="off"
  ENABLE_GE_COV="off"
  PLATFORM=""
  PRODUCT="normal"
  ENABLE_GITEE="off"
  LUOJIANET_MODE="off"
  # Process the options
  while getopts 'ustchj:p:g:vS:M' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      u)
        # ENABLE_GE_UT_ONLY_COMPILE="on"
        ENABLE_GE_UT="on"
        ;;
      s)
        ENABLE_GE_ST="on"
        ;;
      t)
        ENABLE_GE_UT="on"
        ;;
      c)
        ENABLE_GE_COV="on"
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
      p)
        PLATFORM=$OPTARG
        ;;
      g)
        PRODUCT=$OPTARG
        ;;
      S)
        check_on_off $OPTARG S
        ENABLE_GITEE="$OPTARG"
        echo "enable download from gitee"
        ;;
      M)
        LUOJIANET_MODE="on"
        ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
    esac
  done
}
checkopts "$@"

git submodule update --init metadef
git submodule update --init parser

mk_dir() {
    local create_dir="$1"  # the target to make

    mkdir -pv "${create_dir}"
    echo "created ${create_dir}"
}

# GraphEngine build start
echo "---------------- GraphEngine build start ----------------"

# create build path
build_graphengine()
{
  echo "create build directory and build GraphEngine";
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"

  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH"

  if [[ "X$ENABLE_GE_COV" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GE_COV=ON"
  fi

  if [[ "X$ENABLE_GE_UT" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GE_UT=ON"
  fi

  if [[ "X$ENABLE_GE_ST" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GE_ST=ON"
  fi

  if [[ "X$ENABLE_GITEE" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GITEE=ON"
  fi

  if [[ "X$LUOJIANET_MODE" = "Xoff" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_OPEN_SRC=True -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH} -DPLATFORM=${PLATFORM} -DPRODUCT=${PRODUCT}"
  else
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_D=ON -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH}"
  fi

  echo "${CMAKE_ARGS}"
  cmake ${CMAKE_ARGS} ..
  if [ $? -ne 0 ]
  then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    return 1
  fi
  COMMON_TARGET="ge_local_engine ge_local_opskernel_builder ge_common engine fmk_parser parser_common _caffe_parser fmk_onnx_parser graph register engine_conf.json optimizer_priority.pbtxt "
  TARGET=${COMMON_TARGET}
  if [ "x${PLATFORM}" = "xtrain" ]
  then
    TARGET="ge_runner fwk_atc.bin ${TARGET}"
  elif [ "x${PLATFORM}" = "xinference" ]
  then
    TARGET="ge_compiler atc_atc.bin ge_executor_shared ${TARGET}"
  elif [ "X$ENABLE_GE_ST" = "Xon" ]
  then
    TARGET="ge_graph_dsl_test ge_running_env_test graph_engine_test"
  elif [ "X$ENABLE_GE_UT" = "Xon" ]
  then
    TARGET="ut_libgraph ut_libge_multiparts_utest ut_libge_others_utest ut_libge_kernel_utest ut_libge_distinct_load_utest"
  elif [ "X$LUOJIANET_MODE" = "Xon" ]
  then
    TARGET="ge_common graph"
  elif [ "x${PLATFORM}" = "xall" ]
  then
    # build all the target
    TARGET="ge_runner ge_compiler fwk_atc.bin atc_atc.bin ge_executor_shared ${TARGET}"
  fi

  make ${VERBOSE} ${TARGET} -j${THREAD_NUM} && make install
  if [ $? -ne 0 ]
  then
    echo "execute command: make ${VERBOSE} -j${THREAD_NUM} && make install failed."
    return 1
  fi
  echo "GraphEngine build success!"
}
g++ -v
mk_dir ${OUTPUT_PATH}
build_graphengine || { echo "GraphEngine build failed."; return; }
echo "---------------- GraphEngine build finished ----------------"
rm -f ${OUTPUT_PATH}/libgmock*.so
rm -f ${OUTPUT_PATH}/libgtest*.so
rm -f ${OUTPUT_PATH}/lib*_stub.so

chmod -R 750 ${OUTPUT_PATH}
find ${OUTPUT_PATH} -name "*.so*" -print0 | xargs -0 chmod 500

echo "---------------- GraphEngine output generated ----------------"

if [[ "X$ENABLE_GE_UT" = "Xon" || "X$ENABLE_GE_COV" = "Xon" ]]; then
    cp ${BUILD_PATH}/tests/ut/common/graph/ut_libgraph ${OUTPUT_PATH}
    cp ${BUILD_PATH}/tests/ut/ge/ut_libge_multiparts_utest ${OUTPUT_PATH}
    cp ${BUILD_PATH}/tests/ut/ge/ut_libge_distinct_load_utest ${OUTPUT_PATH}
    cp ${BUILD_PATH}/tests/ut/ge/ut_libge_others_utest ${OUTPUT_PATH}
    cp ${BUILD_PATH}/tests/ut/ge/ut_libge_kernel_utest ${OUTPUT_PATH}

    RUN_TEST_CASE=${OUTPUT_PATH}/ut_libgraph && ${RUN_TEST_CASE} &&
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_libge_multiparts_utest && ${RUN_TEST_CASE} &&
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_libge_distinct_load_utest && ${RUN_TEST_CASE} &&
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_libge_others_utest && ${RUN_TEST_CASE} &&
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_libge_kernel_utest && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi
    echo "Generating coverage statistics, please wait..."
    cd ${BASEPATH}
    rm -rf ${BASEPATH}/cov
    mkdir ${BASEPATH}/cov
    lcov -c -d build/tests/ut/ge -d build/tests/ut/common/graph/ -o cov/tmp.info
    lcov -r cov/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/tests/*' '/usr/local/*' '/usr/include/*' '*/metadef/*' '*/parser/*' -o cov/coverage.info
    cd ${BASEPATH}/cov
    genhtml coverage.info
fi

if [[ "X$ENABLE_GE_ST" = "Xon" ]]; then
    #prepare engine & opskernel so
    mkdir -p ${OUTPUT_PATH}/plugin/nnengine
    mkdir -p ${OUTPUT_PATH}/plugin/nnengine/ge_config
    mkdir -p ${OUTPUT_PATH}/plugin/opskernel
    cp ${BUILD_PATH}/tests/framework/libnnengine.so ${OUTPUT_PATH}/plugin/nnengine
    cp ${BUILD_PATH}/engine_conf.json ${OUTPUT_PATH}/plugin/nnengine/ge_config
    cp ${BUILD_PATH}/tests/framework/libge_local_engine.so ${OUTPUT_PATH}/plugin/opskernel
    #prepare st execution bin
    cp ${BUILD_PATH}/tests/st/testcase/graph_engine_test ${OUTPUT_PATH}
    cp ${BUILD_PATH}/tests/framework/ge_running_env/tests/ge_running_env_test ${OUTPUT_PATH}
    cp ${BUILD_PATH}/tests/framework/ge_graph_dsl/tests/ge_graph_dsl_test ${OUTPUT_PATH}
    #execute st testcase
    RUN_TEST_CASE=${OUTPUT_PATH}/ge_running_env_test && ${RUN_TEST_CASE}
    RUN_TEST_CASE=${OUTPUT_PATH}/graph_engine_test && ${RUN_TEST_CASE}
    RUN_TEST_CASE=${OUTPUT_PATH}/ge_graph_dsl_test && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! ST FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi
    # remove plugin
    rm -rf ${OUTPUT_PATH}/plugin
fi

# generate output package in tar form, including ut/st libraries/executables
generate_package()
{
  cd "${BASEPATH}"

  GRAPHENGINE_LIB_PATH="lib"
  ACL_PATH="acllib/lib64"
  FWK_PATH="fwkacllib/lib64"
  ATC_PATH="atc/lib64"
  ATC_BIN_PATH="atc/bin"
  FWK_BIN_PATH="fwkacllib/bin"
  FWK_INCLUDE_PATH="fwkacllib/include"
  ATC_INCLUDE_PATH="atc/include"
  NNENGINE_PATH="plugin/nnengine/ge_config"
  OPSKERNEL_PATH="plugin/opskernel"

  ACL_LIB=("libge_common.so" "libgraph.so" "libregister.so" "liberror_manager.so" "libge_executor.so")
  ATC_LIB=("libc_sec.so" "libge_common.so" "libge_compiler.so" "libgraph.so" "libregister.so" "liberror_manager.so")
  FWK_LIB=("libge_common.so" "libge_runner.so" "libgraph.so" "libregister.so" "liberror_manager.so")
  PLUGIN_OPSKERNEL=("libge_local_engine.so" "libge_local_opskernel_builder.so" "optimizer_priority.pbtxt")
  PARSER_LIB=("lib_caffe_parser.so" "libfmk_onnx_parser.so" "libfmk_parser.so" "libparser_common.so")

  rm -rf ${OUTPUT_PATH:?}/${FWK_PATH}/
  rm -rf ${OUTPUT_PATH:?}/${ACL_PATH}/
  rm -rf ${OUTPUT_PATH:?}/${ATC_PATH}/
  rm -rf ${OUTPUT_PATH:?}/${ATC_BIN_PATH}/
  rm -rf ${OUTPUT_PATH:?}/${FWK_BIN_PATH}/

  mk_dir "${OUTPUT_PATH}/${FWK_PATH}/${NNENGINE_PATH}"
  mk_dir "${OUTPUT_PATH}/${FWK_PATH}/${OPSKERNEL_PATH}"
  mk_dir "${OUTPUT_PATH}/${ATC_PATH}/${NNENGINE_PATH}"
  mk_dir "${OUTPUT_PATH}/${ATC_PATH}/${OPSKERNEL_PATH}"
  mk_dir "${OUTPUT_PATH}/${ACL_PATH}"
  mk_dir "${OUTPUT_PATH}/${ATC_BIN_PATH}"
  mk_dir "${OUTPUT_PATH}/${FWK_BIN_PATH}"
  mk_dir "${OUTPUT_PATH}/${FWK_INCLUDE_PATH}"
  mk_dir "${OUTPUT_PATH}/${ATC_INCLUDE_PATH}"

  cd "${OUTPUT_PATH}"

  find ./ -name graphengine_lib.tar -exec rm {} \;

  cp ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH}/engine_conf.json ${OUTPUT_PATH}/${FWK_PATH}/${NNENGINE_PATH}
  cp ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH}/engine_conf.json ${OUTPUT_PATH}/${ATC_PATH}/${NNENGINE_PATH}

  find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name libengine.so -exec cp -f {} ${OUTPUT_PATH}/${FWK_PATH}/${NNENGINE_PATH}/../ \;
  find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name libengine.so -exec cp -f {} ${OUTPUT_PATH}/${ATC_PATH}/${NNENGINE_PATH}/../ \;

  MAX_DEPTH=1
#  if [ "x${PLATFORM}" = "xall" ] || [ "x${PLATFORM}" = "xinference" ]
#  then
#    MAX_DEPTH=2
#  fi
  for lib in "${PLUGIN_OPSKERNEL[@]}";
  do
    find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth ${MAX_DEPTH} -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${FWK_PATH}/${OPSKERNEL_PATH} \;
    find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth ${MAX_DEPTH} -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${ATC_PATH}/${OPSKERNEL_PATH} \;
  done

  for lib in "${PARSER_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${FWK_PATH} \;
    find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${ATC_PATH} \;
  done

  for lib in "${FWK_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${FWK_PATH} \;
  done

  for lib in "${ACL_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${ACL_PATH} \;
  done

  for lib in "${ATC_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${GRAPHENGINE_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${ATC_PATH} \;
  done

  find ./lib/atclib -name atc.bin -exec cp {} "${OUTPUT_PATH}/${ATC_BIN_PATH}" \;
  find ./lib/fwkacl -name atc.bin -exec cp {} "${OUTPUT_PATH}/${FWK_BIN_PATH}" \;

  cp -r ${OUTPUT_PATH}/../metadef/inc/external/* ${ATC_INCLUDE_PATH}
  cp -r ${OUTPUT_PATH}/../parser/inc/external/* ${ATC_INCLUDE_PATH}
  cp -r ${OUTPUT_PATH}/../inc/external/* ${ATC_INCLUDE_PATH}

  cp -r ${OUTPUT_PATH}/../metadef/inc/external/* ${FWK_INCLUDE_PATH}
  cp -r ${OUTPUT_PATH}/../parser/inc/external/* ${FWK_INCLUDE_PATH}
  cp -r ${OUTPUT_PATH}/../inc/external/* ${FWK_INCLUDE_PATH}

  if [ "x${PLATFORM}" = "xtrain" ]
  then
    tar -zcf graphengine_lib.tar fwkacllib
  elif [ "x${PLATFORM}" = "xinference" ]
  then
    tar -zcf graphengine_lib.tar acllib atc
  elif [ "x${PLATFORM}" = "xall" ]
  then
    tar -zcf graphengine_lib.tar fwkacllib acllib atc
  fi
}

if [[ "X$ENABLE_GE_UT" = "Xoff" && "X$ENABLE_GE_ST" = "Xoff" && "X$LUOJIANET_MODE" = "Xoff" ]]; then
  generate_package
elif [ "X$LUOJIANET_MODE" = "Xon" ]
then
  cd "${OUTPUT_PATH}"
  find ./ -name graphengine_lib.tar -exec rm {} \;
  tar -zcf graphengine_lib.tar lib
fi
echo "---------------- GraphEngine package archive generated ----------------"
