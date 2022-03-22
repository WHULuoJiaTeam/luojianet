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
BUILD_PATH=$BASEPATH/../../build/
OUTPUT_PATH=$BASEPATH/../../output/

echo $BUILD_PATH

export LD_LIBRARY_PATH=/usr/local/HiAI/driver/lib64:${BUILD_PATH}../third_party/prebuild/x86_64/:/usr/local/HiAI/runtime/lib64:${BUILD_PATH}/graphengine/:${D_LINK_PATH}/x86_64/:${LD_LIBRARY_PATH}
echo ${LD_LIBRARY_PATH}
${OUTPUT_PATH}/ut_libgraph &&
${OUTPUT_PATH}/ut_libge_multiparts_utest &&
${OUTPUT_PATH}/ut_libge_distinct_load_utest &&
${OUTPUT_PATH}/ut_libge_others_utest &&
${OUTPUT_PATH}/ut_libge_kernel_utest
