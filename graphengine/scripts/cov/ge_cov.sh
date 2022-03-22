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


function help(){
    cat <<-EOF
Usage: ge cov [OPTIONS]

Options:
    -a, --all          Full coverage
    -i, --increment    Increment coverage
    -d, --directory    Coverage of directory
    -h, --help 
EOF

}

PROJECT_HOME=${PROJECT_HOME:-$(dirname "$0")/../../}
PROJECT_HOME=$(cd $PROJECT_HOME || return; pwd)

ALL_COV_GEN_PATH=${PROJECT_HOME}/cov/all
DIFF_FILE_PATH=${PROJECT_HOME}/cov/diff
DIFF_FILE_NAME=${DIFF_FILE_PATH}/inc_change_diff.txt

function process_diff_format(){
    sed -i "s/--- a/--- \/code\/Turing\/graphEngine/g" ${DIFF_FILE_NAME}
    sed -i "s/+++ b/+++ \/code\/Turing\/graphEngine/g" ${DIFF_FILE_NAME}
}


function add_cov_generate(){
    addlcov --diff ${ALL_COV_GEN_PATH}/coverage.info ${DIFF_FILE_NAME} -o ${PROJECT_HOME}/cov/diff/inc_coverage.info
}

function gen_add_cov_html(){
    genhtml --prefix ${PROJECT_HOME} -o ${PROJECT_HOME}/cov/diff/html ${PROJECT_HOME}/cov/diff/inc_coverage.info --legend -t CHG --no-branch-coverage --no-function-coverage
}

function increment_cov_for_directory(){
    [ -n "${DIFF_FILE_PATH}" ] && rm -rf "${DIFF_FILE_PATH}"
    mkdir -p ${DIFF_FILE_PATH}
    git diff HEAD -- $1 >>${DIFF_FILE_NAME}
    process_diff_format
    add_cov_generate
    gen_add_cov_html
}

function run_all_coverage(){
    [ -n "${ALL_COV_GEN_PATH}" ] && rm -rf ${ALL_COV_GEN_PATH}
    mkdir -p ${ALL_COV_GEN_PATH}
    pushd "${PWD}" >/dev/null
        cd ${PROJECT_HOME} 
        lcov -c -d build/tests/ut/ge -d build/tests/ut/common/graph/ -o ${ALL_COV_GEN_PATH}/tmp.info
        lcov -r ${ALL_COV_GEN_PATH}/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/tests/*' '/usr/local/*' '/usr/include/*' '*/metadef/*' '*/parser/*' -o ${ALL_COV_GEN_PATH}/coverage.info
        cd ${ALL_COV_GEN_PATH}
        genhtml coverage.info
    popd  >/dev/null
}

function do_coverage_run(){
    local cov_mode=$1
    local directory_dir=$2
    
    run_all_coverage

    if [ "$cov_mode" = "all" ]; then
        exit 1
    elif [ -n "$directory_dir" ]; then
        increment_cov_for_directory $directory_dir
    else 
        increment_cov_for_directory "ge"
    fi 
}

function parse_args(){
    parsed_args=$(getopt -a -o aid::h --long all,increment,directory::,help -- "$@") || {
        help
        exit 1
    }

    if [ $# -lt 1 ]; then
        run_all_coverage
        exit 1
    fi

    local cov_mode="increment"
    local directory_dir=
    eval set -- "$parsed_args"
    while true; do
        case "$1" in
            -a | --all)
                cov_mode="all"
                ;;
            -i | --increment)
                ;;
            -d | --directory)
                directory_dir=$2
                shift
                ;;
            -h | --help)
                help; exit 1;
                ;;
            --)
                shift; break;
                ;;
            *)
                help; exit 1;
                ;;
        esac
        shift 
    done
    do_coverage_run $cov_mode $directory_dir
}

function main(){
    parse_args "$@"
}

main "$@"

set +e