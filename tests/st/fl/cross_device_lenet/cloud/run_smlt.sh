#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

export PYTHONPATH=../../../../:$PYTHONPATH
server_num=$1
worker_num=$2
ip=$3
port=$4

for((i=0;i<worker_num;i++));
do
  ofs=`expr $i % $server_num`
  real_port=`expr $port + $ofs`
  echo $real_port
  python simulator.py --pid=$i --http_ip=$ip --http_port=$port --use_elb=True --server_num=$1 > simulator_$i.log 2>&1 &
done
