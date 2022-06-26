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
import os
import pytest
from mindspore import context


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_hccl_allreduce():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system("mpirun --allow-run-as-root -n 8 pytest -s test_allreduce.py")
    assert return_code == 0
