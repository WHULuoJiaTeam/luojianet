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

"""SigmoidGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

sigmoid_grad_op_info = TBERegOp("SigmoidGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("sigmoid_grad.so") \
    .compute_cost(10) \
    .kernel_name("sigmoid_grad") \
    .partial_flag(True) \
    .input(0, "x", False, "required", "all") \
    .input(1, "y", False, "required", "all") \
    .output(0, "z", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(sigmoid_grad_op_info)
def _sigmoid_grad_tbe():
    """SigmoidGrad TBE register"""
    return
