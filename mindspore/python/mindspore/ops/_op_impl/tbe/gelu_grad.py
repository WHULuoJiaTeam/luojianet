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

"""GeLUGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

gelu_grad_op_info = TBERegOp("GeLUGrad") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("gelu_grad.so") \
    .compute_cost(10) \
    .kernel_name("gelu_grad") \
    .partial_flag(True) \
    .input(0, "dy", False, "required", "all") \
    .input(1, "x", False, "required", "all") \
    .input(2, "y", False, "required", "all") \
    .output(0, "z", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .dtype_format(DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ) \
    .get_op_info()


@op_info_register(gelu_grad_op_info)
def _gelu_grad_tbe():
    """GeLUGrad TBE register"""
    return
