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

"""EnvironGet op"""
from luojianet_ms.ops.op_info_register import op_info_register, AiCPURegOp, DataType

environ_get_op_info = AiCPURegOp("EnvironGet") \
    .fusion_type("OPAQUE") \
    .attr("value_type", "int") \
    .input(0, "env", "required") \
    .input(1, "key", "required") \
    .input(2, "default", "required") \
    .output(0, "value", "required") \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.U32_Default, DataType.U32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.U16_Default, DataType.U16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.U64_Default, DataType.U64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.BOOL_Default, DataType.BOOL_Default) \
    .get_op_info()

@op_info_register(environ_get_op_info)
def _environ_get_aicpu():
    """EnvironGet AiCPU register"""
    return
