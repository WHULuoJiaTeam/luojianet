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

"""PReLUGrad op"""
from luojianet_ms.ops.op_info_register import op_info_register, TBERegOp, DataType

prelu_grad_op_info = TBERegOp("PReLUGrad") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("prelu_grad.so") \
    .compute_cost(10) \
    .kernel_name("prelu_grad") \
    .partial_flag(True) \
    .input(0, "grads", False, "required", "all") \
    .input(1, "features", False, "required", "all") \
    .input(2, "weights", False, "required", "all") \
    .output(0, "dx", False, "required", "all") \
    .output(1, "da", False, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None, DataType.None_None,
                  DataType.None_None) \
    .get_op_info()


@op_info_register(prelu_grad_op_info)
def _prelu_grad_tbe():
    """PReLUGrad TBE register"""
    return
