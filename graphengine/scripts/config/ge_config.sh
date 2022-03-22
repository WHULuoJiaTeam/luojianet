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

function help(){
    cat <<-EOF
Usage: ge config [OPTIONS]

update server config for ge, you need input all config info (ip, user, password)

Options:
    -i, --ip           Config ip config
    -u, --user         Config user name
    -p, --password     Config password
    -h, --help

Example: ge config -i=121.36.**.** -u=asc**, -p=Asc***\#@\!\$     (Need add escape character \ before special charactor $、#、!)

EOF

}

function write_config_file(){
    local IP=$1
    local USER=$2
    local PASSWORD=$3
    if [[ -z "$IP" ]] || [[ -z "$USER" ]] || [[ -z "$USER" ]]; then
        echo "You need input all info （ip, user,password）obout server config !!!"
        help
        exit 1
    fi

    local PASSWORD=${PASSWORD//!/\\!}
    local PASSWORD=${PASSWORD//#/\\#}
    local PASSWORD=${PASSWORD/\$/\\\$}
    local SERVER_CONFIG_FILE=${PROJECT_HOME}/scripts/config/server_config.sh
    [ -n "${SERVER_CONFIG_FILE}" ] && rm -rf "${SERVER_CONFIG_FILE}"

    cat>${SERVER_CONFIG_FILE}<<-EOF
SERVER_PATH=http://${IP}/package/etrans
DEP_USER=${USER}
DEP_PASSWORD=${PASSWORD}

EOF

}



function parse_args(){
    parsed_args=$(getopt -a -o i::u::p::h --long ip::,user::,password::,help -- "$@") || {
        help
        exit 1
    }

    if [ $# -lt 1 ]; then
        help
        exit 1
    fi
    local IP=
    local USER=
    local PASSWORD=

    eval set -- "$parsed_args"
    while true; do
        case "$1" in
            -i | --ip)
                IP=$2
                ;;
            -u | --user)
                USER=$2
                ;; 
            -p | --password)
                PASSWORD=$2
                ;;
            -h | --help)
                help; exit;
                ;;
            --)
                shift; break;
                ;;
            *)
                help; exit 1
                ;;
        esac
        shift 2
    done

    write_config_file $IP $USER $PASSWORD
}

function main(){
    parse_args "$@"
}

main "$@"

set +e