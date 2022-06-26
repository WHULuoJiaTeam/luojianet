#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
BASE_PATH=$(cd "$(dirname $0)"; pwd)
CONFIG_PATH=/home/workspace/mindspore_config
export DEVICE_NUM=8
export RANK_SIZE=$DEVICE_NUM
source ${BASE_PATH}/env.sh
unset SLOG_PRINT_TO_STDOUT
export MINDSPORE_HCCL_CONFIG_PATH=$CONFIG_PATH/hccl/rank_table_${DEVICE_NUM}p.json
export LD_LIBRARY_PATH=/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=/usr/local/Ascend/opp/

process_pid=()
for((i=0; i<$DEVICE_NUM; i++)); do
    rm -rf ${BASE_PATH}/parallel_strategy_search${i}
    mkdir ${BASE_PATH}/parallel_strategy_search${i}
    cp -r ${BASE_PATH}/parallel_strategy_search.py  ${BASE_PATH}/parallel_strategy_search${i}/
    cd ${BASE_PATH}/parallel_strategy_search${i}
    export RANK_ID=${i}
    export DEVICE_ID=${i}
    echo "start training for device $i"
    env > env$i.log
    pytest -s -v parallel_strategy_search.py::test_auto_parallel_strategy_search_axis_1_basic > parallel_strategy_search$i.log 2>&1 &
    process_pid[${i}]=`echo $!`
done

for((i=0; i<${DEVICE_NUM}; i++)); do
    wait ${process_pid[i]}
    status=`echo $?`
    if [ "${status}" != "0" ]; then
        echo "[ERROR] test_parallel_strategy_search failed. status: ${status}"
        exit 1
    else
        echo "[INFO] test_parallel_strategy_search success."
    fi
done

exit 0
