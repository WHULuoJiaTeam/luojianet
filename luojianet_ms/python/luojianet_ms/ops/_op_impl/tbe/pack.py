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

"""Pack op"""
from luojianet_ms.ops.op_info_register import op_info_register, TBERegOp, DataType

stack_op_info = TBERegOp("Stack") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("pack.so") \
    .compute_cost(10) \
    .kernel_name("pack") \
    .partial_flag(True) \
    .need_check_supported(False) \
    .attr("axis", "optional", "int", "all") \
    .input(0, "x", False, "dynamic", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.U16_Default, DataType.U16_Default) \
    .dtype_format(DataType.U32_Default, DataType.U32_Default) \
    .dtype_format(DataType.U64_Default, DataType.U64_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.BOOL_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I8_NDHWC, DataType.I8_NDHWC) \
    .dtype_format(DataType.I16_NDHWC, DataType.I16_NDHWC) \
    .dtype_format(DataType.I32_NDHWC, DataType.I32_NDHWC) \
    .dtype_format(DataType.I64_NDHWC, DataType.I64_NDHWC) \
    .dtype_format(DataType.U8_NDHWC, DataType.U8_NDHWC) \
    .dtype_format(DataType.U16_NDHWC, DataType.U16_NDHWC) \
    .dtype_format(DataType.U32_NDHWC, DataType.U32_NDHWC) \
    .dtype_format(DataType.U64_NDHWC, DataType.U64_NDHWC) \
    .dtype_format(DataType.F16_NDHWC, DataType.F16_NDHWC) \
    .dtype_format(DataType.F32_NDHWC, DataType.F32_NDHWC) \
    .dtype_format(DataType.BOOL_NDHWC, DataType.BOOL_NDHWC) \
    .get_op_info()


@op_info_register(stack_op_info)
def _pack_tbe():
    """Pack TBE register"""
    return
