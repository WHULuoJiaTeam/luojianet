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

PROJECT_HOME=${PROJECT_HOME:-$(dirname "$0")/../../}
PROJECT_HOME=$(cd $PROJECT_HOME || return; pwd)
DOWNLOAD_PATH=${PROJECT_HOME}/deps
DEP_LIB_DIR=./lib
DEP_TMP_DIR=./tmp

function extract_deps_so()
{
    echo "begin to extract .run file ........."
    chmod 777 ./${DRIVER_RUN_NAME}
    unzip ${DEV_TOOLS_PACKAGE}.zip
    chmod -R 777 ${DEV_TOOLS_PACKAGE}
    [ -n "${DEP_TMP_DIR}" ] && rm -rf "${DEP_TMP_DIR}"
    ./${DRIVER_RUN_NAME} --noexec --extract=${DEP_TMP_DIR}/driver
    ./${DEV_TOOLS_PACKAGE}/${ATC_RUN_NAME} --noexec --extract=${DEP_TMP_DIR}/atc
    ./${DEV_TOOLS_PACKAGE}/${ACL_RUN_NAME} --noexec --extract=${DEP_TMP_DIR}/acllib
    ./${DEV_TOOLS_PACKAGE}/${FWKACL_RUN_NAME} --noexec --extract=${DEP_TMP_DIR}/fwkacllib
}

function extract_deps_so_community()
{
    echo "begin to extract .run file ........."
    chmod +x ./${DRIVER_RUN_NAME_C}
    chmod +x ./${PACKAGE_NAME_C}
    [ -n "${DEP_TMP_DIR}" ] && rm -rf "${DEP_TMP_DIR}"
    ./${DRIVER_RUN_NAME_C} --noexec --extract=${DEP_TMP_DIR}/driver
    ./${PACKAGE_NAME_C} --noexec --extract=${DEP_TMP_DIR}/Packages_tmp
    ${DEP_TMP_DIR}/Packages_tmp/run_package/${ATC_RUN_NAME_C} --noexec --extract=${DEP_TMP_DIR}/atc
    ${DEP_TMP_DIR}/Packages_tmp/run_package/${ACL_RUN_NAME_C} --noexec --extract=${DEP_TMP_DIR}/acllib
    ${DEP_TMP_DIR}/Packages_tmp/run_package/${FWKACL_RUN_NAME_C} --noexec --extract=${DEP_TMP_DIR}/fwkacllib
}

function copy_so_to_target_dir()
{
    mkdir -p $DEP_LIB_DIR
    mv ${DEP_TMP_DIR}/driver/driver $DEP_LIB_DIR/driver
    mv ${DEP_TMP_DIR}/atc/atc $DEP_LIB_DIR/atc
    mv ${DEP_TMP_DIR}/acllib/acllib $DEP_LIB_DIR/acllib
    mv ${DEP_TMP_DIR}/fwkacllib/fwkacllib $DEP_LIB_DIR/fwkacllib
}

function clear_libs()
{
    [ -n "${DOWNLOAD_PATH}" ] && rm -rf "${DOWNLOAD_PATH}"
}

function download_runs()
{
    source scripts/update/deps_config.sh
    echo "begin to download .run file ........."
    clear_libs
    mkdir -p ./ ${DOWNLOAD_PATH}
    pushd "${DOWNLOAD_PATH}" >/dev/null
        cd ${DOWNLOAD_PATH} 
        wget --user=${DEP_USER} --password=${DEP_PASSWORD}  ${DRIVER_URL}
        wget --user=${DEP_USER} --password=${DEP_PASSWORD}  ${DEV_TOOLS_URL}
    popd >/dev/null
}

function download_runs_from_community
{
    source scripts/update/deps_config_community.sh
    echo "begin to download .run file from community........."
    clear_libs
    mkdir -p ./ ${DOWNLOAD_PATH}
    pushd "${DOWNLOAD_PATH}" >/dev/null
        cd ${DOWNLOAD_PATH} 
        wget ${DRIVER_URL_C}
        wget ${PACKAGE_URL_C}
    popd >/dev/null
}

function install_deps()
{
    source scripts/update/deps_config.sh
    mkdir -p ./ ${DOWNLOAD_PATH}
    pushd "${DOWNLOAD_PATH}" >/dev/null
        cd ${DOWNLOAD_PATH}
        extract_deps_so
        copy_so_to_target_dir
    popd >/dev/null
}

function install_deps_community()
{
    source scripts/update/deps_config_community.sh
    mkdir -p ./ ${DOWNLOAD_PATH}
    pushd "${DOWNLOAD_PATH}" >/dev/null
        cd ${DOWNLOAD_PATH}
        extract_deps_so_community
        copy_so_to_target_dir
    popd >/dev/null
}


function help(){
    cat <<-EOF
Usage: ge update [OPTIONS]

update dependencies of build and test

Options:
    -p, --public       Download dependencies from community
    -d, --download     Download dependencies
    -i, --install      Install dependencies
    -c, --clear        Clear dependencies 
    -h, --help
EOF

}

function parse_args(){
    parsed_args=$(getopt -a -o pdich --long public,download,install,clear,help -- "$@") || {
        help
        exit 1
    }

    if [ $# -lt 1 ]; then
        download_runs_from_community
        install_deps_community
        exit 1
    fi

    eval set -- "$parsed_args"
    while true; do
        case "$1" in
            -p | --public)
                download_runs_from_community
                install_deps_community
                ;;
            -d | --download)
                download_runs
                ;;
            -i | --install)
                install_deps
                ;; 
            -c | --clear)
                clear_libs
                ;;
            -h | --help)
                help; exit 1;
                ;;
            --)
                shift; break;
                ;;
            *)
                help; exit 1
                ;;
        esac
        shift
    done
}

function main(){
    parse_args "$@"
}

main "$@"

set +e