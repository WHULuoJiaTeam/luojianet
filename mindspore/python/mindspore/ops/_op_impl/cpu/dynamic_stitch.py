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

"""DynamicStitch op"""
from mindspore.ops.op_info_register import op_info_register, CpuRegOp, DataType

dynamic_stitch_op_info = CpuRegOp("DynamicStitch") \
    .input(0, "indices", "dynamic") \
    .input(1, "data", "dynamic") \
    .output(0, "y", "required") \
    .dtype_format(DataType.I32_Default, DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.I32_Default, DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.I32_Default, DataType.U16_Default, DataType.U16_Default) \
    .dtype_format(DataType.I32_Default, DataType.U32_Default, DataType.U32_Default) \
    .dtype_format(DataType.I32_Default, DataType.U64_Default, DataType.U64_Default) \
    .dtype_format(DataType.I32_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.I32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I32_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.I32_Default, DataType.BOOL_Default, DataType.BOOL_Default) \
    .get_op_info()


@op_info_register(dynamic_stitch_op_info)
def _dynamic_stitch_cpu():
    """DynamicStitch CPU register"""
    return
