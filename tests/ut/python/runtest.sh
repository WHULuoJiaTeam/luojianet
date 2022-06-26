#!/bin/bash
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
CURRPATH=$(cd "$(dirname $0)" || exit; pwd)
IGNORE_EXEC="--ignore=$CURRPATH/exec"

source $CURRPATH/env.sh

if [ $# -eq 1 ]  &&  ([ "$1" == "stage1" ] || [ "$1" == "stage2" ] || [ "$1" == "stage3" ] || [ "$1" == "stage4" ]); then
    if [ $1 == "stage1" ]; then
        echo "run python dataset ut"
        pytest -v $CURRPATH/dataset

        RET=$?
        if [ ${RET} -ne 0 ]; then
            exit ${RET}
        fi

        echo "run python debugger gpu ut"
        pytest -v $CURRPATH/debugger/gpu_tests

        RET=$?
        if [ ${RET} -ne 0 ]; then
            exit ${RET}
        fi

    elif [ $1 == "stage2" ]; then
        echo "run python parallel"
        pytest -s $CURRPATH/parallel/*.py

        RET=$?
        if [ ${RET} -ne 0 ]; then
            exit ${RET}
        fi

    elif [ $1 == "stage3" ]; then
        echo "run python ops, pynative_mode, pipeline, train ut"
        pytest -v $CURRPATH/ops $CURRPATH/pynative_mode

        RET=$?
        if [ ${RET} -ne 0 ]; then
            exit ${RET}
        fi

        pytest -v $CURRPATH/pipeline $CURRPATH/train
        RET=$?
        if [ ${RET} -ne 0 ]; then
            exit ${RET}
        fi

    elif [ $1 == "stage4" ]; then
        echo "run ut"
        pytest -v $CURRPATH/nn

        RET=$?
        if [ ${RET} -ne 0 ]; then
            exit ${RET}
        fi

        pytest -v --ignore=$CURRPATH/dataset --ignore=$CURRPATH/debugger/gpu_tests --ignore=$CURRPATH/parallel --ignore=$CURRPATH/ops --ignore=$CURRPATH/pynative_mode --ignore=$CURRPATH/pipeline --ignore=$CURRPATH/train --ignore=$CURRPATH/nn $IGNORE_EXEC $CURRPATH

        RET=$?
        if [ ${RET} -ne 0 ]; then
            exit ${RET}
        fi
    fi
else
    echo "run all python ut"
    pytest $CURRPATH/dataset
    RET=$?
    if [ ${RET} -ne 0 ]; then
        exit ${RET}
    fi

    pytest $CURRPATH/debugger/gpu_tests
    RET=$?
    if [ ${RET} -ne 0 ]; then
        exit ${RET}
    fi

    pytest -v $CURRPATH/parallel/*.py
    RET=$?
    if [ ${RET} -ne 0 ]; then
        exit ${RET}
    fi

    pytest -v $CURRPATH/ops $CURRPATH/pynative_mode
    RET=$?
    if [ ${RET} -ne 0 ]; then
        exit ${RET}
    fi

    pytest -v $CURRPATH/pipeline $CURRPATH/train
    RET=$?
    if [ ${RET} -ne 0 ]; then
        exit ${RET}
    fi

    pytest -v $CURRPATH/nn
    RET=$?
    if [ ${RET} -ne 0 ]; then
        exit ${RET}
    fi

    pytest -v --ignore=$CURRPATH/dataset --ignore=$CURRPATH/debugger/gpu_tests --ignore=$CURRPATH/parallel --ignore=$CURRPATH/ops --ignore=$CURRPATH/pynative_mode --ignore=$CURRPATH/pipeline --ignore=$CURRPATH/train --ignore=$CURRPATH/nn $IGNORE_EXEC $CURRPATH
    RET=$?
    if [ ${RET} -ne 0 ]; then
        exit ${RET}
    fi
fi

RET=$?
exit ${RET}
