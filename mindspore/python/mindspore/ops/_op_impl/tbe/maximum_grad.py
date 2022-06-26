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

"""MaximumGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

maximum_grad_op_info = TBERegOp("MaximumGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("maximum_grad.so") \
    .compute_cost(10) \
    .kernel_name("maximum_grad") \
    .partial_flag(True) \
    .attr("grad_x", "optional", "bool", "all", "true") \
    .attr("grad_y", "optional", "bool", "all", "true") \
    .input(0, "grads", False, "required", "all") \
    .input(1, "x1", False, "required", "all") \
    .input(2, "x2", False, "required", "all") \
    .output(0, "y1", False, "required", "all") \
    .output(1, "y2", False, "required", "all") \
    .op_pattern("broadcast") \
    .dtype_format(DataType.I32_None, DataType.I32_None, DataType.I32_None, DataType.I32_None,
                  DataType.I32_None) \
    .dtype_format(DataType.F16_None, DataType.F16_None, DataType.F16_None, DataType.F16_None,
                  DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None, DataType.F32_None, DataType.F32_None,
                  DataType.F32_None) \
    .get_op_info()


@op_info_register(maximum_grad_op_info)
def _maximum_grad_tbe():
    """MaximumGrad TBE register"""
    return
