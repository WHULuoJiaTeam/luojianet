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
Usage: ge docs [OPTIONS]

Options:
    -b, --brief  Build brief docs
    -a, --all    Build all docs
    -h, --help
EOF

}

PROJECT_HOME=${PROJECT_HOME:-$(dirname "$0")/../../}
PROJECT_HOME=$(cd $PROJECT_HOME || return; pwd)
BRIEF_DOXYFILE_PATH=${PROJECT_HOME}/scripts/docs/Doxyfile_brief
ALL_DOXYFILE_PATH=${PROJECT_HOME}/scripts/docs/Doxyfile_all


function build_brief_docs(){
    rm -rf "${PROJECT_HOME}/docs/doxygen"
    doxygen ${BRIEF_DOXYFILE_PATH}
}

function build_all_docs(){
    rm -rf "${PROJECT_HOME}/docs/doxygen"
    doxygen ${ALL_DOXYFILE_PATH}
}

function parse_args(){
    parsed_args=$(getopt -a -o bah --long brief,all,help -- "$@") || {
        help
        exit 1
    }

    if [ $# -lt 1 ]; then
        build_all_docs
        exit 1
    fi

    eval set -- "$parsed_args"
    while true; do
        case "$1" in
            -b | --brief)
                build_brief_docs
                ;; 
            -a | --all)
                build_all_docs
                ;; 
            -h | --help)
                help;  exit 1;
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
}

function main(){
    parse_args "$@"
}

main "$@"

set +e