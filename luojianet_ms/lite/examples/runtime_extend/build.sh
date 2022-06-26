#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

BASEPATH=$(cd "$(dirname $0)" || exit; pwd)
get_version() {
    VERSION_STR=$(cat ${BASEPATH}/../../../../version.txt)
}
get_version
MODEL_DOWNLOAD_URL="https://download.luojianet_ms.cn/model_zoo/official/lite/quick_start/add_extend.ms"
LUOJIANET_MS_FILE_NAME="luojianet_ms-lite-${VERSION_STR}-linux-x64"
LUOJIANET_MS_FILE="${LUOJIANET_MS_FILE_NAME}.tar.gz"
LUOJIANET_MS_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/LUOJIANET_MS/lite/release/linux/x86_64/${LUOJIANET_MS_FILE}"

mkdir -p build
mkdir -p lib
mkdir -p model
if [ ! -e ${BASEPATH}/model/add_extend.ms ]; then
    wget -c -O ${BASEPATH}/model/add_extend.ms --no-check-certificate ${MODEL_DOWNLOAD_URL}
fi
if [ ! -e ${BASEPATH}/build/${LUOJIANET_MS_FILE} ]; then
  wget -c -O ${BASEPATH}/build/${LUOJIANET_MS_FILE} --no-check-certificate ${LUOJIANET_MS_LITE_DOWNLOAD_URL}
fi
tar -xzf ${BASEPATH}/build/${LUOJIANET_MS_FILE} -C ${BASEPATH}/build/
cp -r ${BASEPATH}/build/${LUOJIANET_MS_FILE_NAME}/runtime/lib/libluojianet_ms-lite.so ${BASEPATH}/lib/
cp -r ${BASEPATH}/build/${LUOJIANET_MS_FILE_NAME}/runtime/include ${BASEPATH}/
cd ${BASEPATH}/build || exit
cmake ${BASEPATH}
make
