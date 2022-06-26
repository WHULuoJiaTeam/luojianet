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

"""Argmin op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

arg_min_op_info = TBERegOp("Argmin") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("arg_min_d.so") \
    .compute_cost(10) \
    .kernel_name("arg_min_d") \
    .partial_flag(True) \
    .attr("axis", "required", "int", "all") \
    .attr("output_dtype", "optional", "type", "all") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.I32_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default) \
    .get_op_info()


@op_info_register(arg_min_op_info)
def _arg_min_tbe():
    """Argmin TBE register"""
    return
