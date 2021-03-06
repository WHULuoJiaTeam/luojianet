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

"""BitwiseAnd op"""
from luojianet_ms.ops.op_info_register import op_info_register, TBERegOp, DataType

bitwise_and_op_info = TBERegOp("BitwiseAnd") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("bitwise_and.so") \
    .compute_cost(10) \
    .kernel_name("bitwise_and") \
    .partial_flag(True) \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .op_pattern("broadcast") \
    .dtype_format(DataType.I16_None, DataType.I16_None, DataType.I16_None) \
    .dtype_format(DataType.U16_None, DataType.U16_None, DataType.U16_None) \
    .dtype_format(DataType.I32_None, DataType.I32_None, DataType.I32_None) \
    .get_op_info()


@op_info_register(bitwise_and_op_info)
def _bitwise_and_tbe():
    """BitwiseAnd TBE register"""
    return
