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

SERVER_CONFIG_FILE=${PROJECT_HOME}/scripts/config/server_config.sh

[ -e $SERVER_CONFIG_FILE ] || {
    echo "You have not config dependencies account info first !!!!!"
    ${PROJECT_HOME}/scripts/config/ge_config.sh -h
    exit 1;
}

source scripts/config/server_config.sh

CPU_ARCH=ubuntu18.04.x86_64
DRIVER_VERSION=20.2.0
CHIP_NAME=A800-9010
PRODUCT_VERSION=driver_C76_TR5

DRIVER_NAME=npu-driver
DRIVER_RUN_NAME=${CHIP_NAME}-${DRIVER_NAME}_${DRIVER_VERSION}_ubuntu18.04-x86_64.run

DEV_TOOLS_VERSION=1.78.t10.0.b100

export ATC_RUN_NAME=Ascend-atc-${DEV_TOOLS_VERSION}-${CPU_ARCH}.run
export ACL_RUN_NAME=Ascend-acllib-${DEV_TOOLS_VERSION}-${CPU_ARCH}.run
export FWKACL_RUN_NAME=Ascend-fwkacllib-${DEV_TOOLS_VERSION}-${CPU_ARCH}.run

DEV_TOOLS_PACKAGE=x86_ubuntu_os_devtoolset_package

export DRIVER_URL=${SERVER_PATH}/${PRODUCT_VERSION}/${DRIVER_RUN_NAME}
export DEV_TOOLS_URL=${SERVER_PATH}/20210428/${DEV_TOOLS_PACKAGE}.zip

set +e