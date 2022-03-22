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
#社区版
CHIP_NAME_C=A800-9010
DRIVER_VERSION_C=21.0.rc1
DRIVER_RUN_NAME_C=${CHIP_NAME_C}-npu-driver_${DRIVER_VERSION_C}_ubuntu18.04-x86_64.run
DRIVER_SERVER_PATH_C=https://obs-9be7.obs.cn-east-2.myhuaweicloud.com
export DRIVER_URL_C=${DRIVER_SERVER_PATH_C}/turing/resourcecenter/Software/AtlasI/A800-9010%201.0.10/NPU/${DRIVER_RUN_NAME_C}

PACKAGE_VERSION_C=5.0.2.alpha002
PACKAGE_NAME_C=Ascend-cann-toolkit_${PACKAGE_VERSION_C}_linux-x86_64.run
PACKAGE_SERVER_PATH_C=https://mirrors.huaweicloud.com
export PACKAGE_URL_C=${PACKAGE_SERVER_PATH_C}/ascend/autoarchive/CANN/${PACKAGE_VERSION_C}/${PACKAGE_NAME_C}

DEV_TOOLS_VERSION_C=1.78.t12.0.b120
CPU_ARCH_C=linux.x86_64
export ATC_RUN_NAME_C=Ascend-atc-${DEV_TOOLS_VERSION_C}-${CPU_ARCH_C}.run
export ACL_RUN_NAME_C=Ascend-acllib-${DEV_TOOLS_VERSION_C}-${CPU_ARCH_C}.run
export FWKACL_RUN_NAME_C=Ascend-fwkacllib-${DEV_TOOLS_VERSION_C}-${CPU_ARCH_C}.run
set +e