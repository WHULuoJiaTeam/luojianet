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

"""ActsULQInputGrad op"""
from luojianet_ms.ops.op_info_register import op_info_register, TBERegOp, DataType

acts_ulq_input_grad_op_info = TBERegOp("ActsULQInputGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("acts_ulq_input_grad.so") \
    .compute_cost(10) \
    .kernel_name("acts_ulq_input_grad") \
    .partial_flag(True) \
    .input(0, "y_grad", False, "required", "all") \
    .input(1, "clamp_min_mask", False, "required", "all") \
    .input(2, "clamp_max_mask", False, "required", "all") \
    .output(0, "x_grad", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.BOOL_Default, DataType.BOOL_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.BOOL_Default, DataType.BOOL_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(acts_ulq_input_grad_op_info)
def _acts_ulq_input_grad_tbe():
    """ActsULQInputGrad TBE register"""
    return
