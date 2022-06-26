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

"""L2NormalizeGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

l2_normalize_grad_op_info = TBERegOp("L2NormalizeGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("l2_normalize_grad.so") \
    .compute_cost(10) \
    .kernel_name("l2_normalize_grad") \
    .partial_flag(True) \
    .attr("axis", "required", "listInt", "all") \
    .attr("epsilon", "required", "float", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "y", False, "required", "all") \
    .input(2, "dy", False, "required", "all") \
    .output(0, "dx", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(l2_normalize_grad_op_info)
def _l2_normalize_grad_tbe():
    """L2NormalizeGrad TBE register"""
    return
