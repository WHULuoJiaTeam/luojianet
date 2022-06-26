# Copyright 2022 Huawei Technologies Co., Ltd
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

"""ArgMaxWithValue op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

arg_max_with_value_ds_op_info = TBERegOp("ArgMaxWithValue") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("arg_max_with_value.so") \
    .compute_cost(10) \
    .kernel_name("arg_max_with_value") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("axis", "required", "int", "all") \
    .input(0, "x", False, "required", "all") \
    .output(0, "indice", False, "required", "all") \
    .output(1, "values", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(arg_max_with_value_ds_op_info)
def _arg_max_with_value_ds_tbe():
    """ArgMaxWithValue TBE register"""
    return
