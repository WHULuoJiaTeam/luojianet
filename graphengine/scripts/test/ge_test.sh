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

function help(){
    cat <<-EOF
Usage: ge test [OPTIONS]

Options:
    -u, --unit          Run unit Test
    -c, --component     Run component Test
    -h, --help 
EOF

}

function unit_test(){
    ${PROJECT_HOME}/build.sh -u
}

function component_test(){
    ${PROJECT_HOME}/build.sh -s
}

function parse_args(){
    parsed_args=$(getopt -a -o uch --long unit,component,help -- "$@") || {
        help
        exit 1
    }

    if [ $# -lt 1 ]; then
        unit_test
        exit 1
    fi

    eval set -- "$parsed_args"
    while true; do
        case "$1" in
            -u | --unit)
                unit_test
                ;; 
            -c | --component)
                component_test
                ;;
            -h | --help)
                help
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