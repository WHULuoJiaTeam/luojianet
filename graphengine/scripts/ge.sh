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

GE_BASH_HOME=$(dirname "$0")
export PROJECT_HOME=${PROJECT_HOME:-${GE_BASH_HOME}/../}
PROJECT_HOME=$(cd $PROJECT_HOME || return; pwd)

function help(){
        cat <<-EOF
Usage: ge  COMMANDS

Run ge commands

Commands:
    env         Prepare docker env
    config      Config dependencies server
    update      Update dependencies
    format      Format code
    build       Build code
    test        Run test of UT/ST
    cov         Run Coverage  
    docs        Generate documents
    clean       Clean 
EOF

}

function ge_error() {
    echo "Error: $*" >&2
    help
    exit 1
}

function main(){
    if [ $# -eq 0 ]; then
        help; exit 0
    fi

    local cmd=$1
    local shell_cmd=
    shift

    case "$cmd" in
        -h|--help)
            help; exit 0
            ;;
        build)
            shell_cmd=${PROJECT_HOME}/build.sh
            ;;
        *)
            shell_cmd=$GE_BASH_HOME/$cmd/ge_$cmd.sh
            ;;

    esac

    [ -e $shell_cmd ] || {
        ge_error "ge $shell_cmd is not found"
    }

    $shell_cmd "$@"
}

main "$@"

