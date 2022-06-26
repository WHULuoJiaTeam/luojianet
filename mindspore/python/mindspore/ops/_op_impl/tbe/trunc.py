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

"""Trunc op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

trunc_op_info = TBERegOp("Trunc") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("trunc.so") \
    .compute_cost(10) \
    .kernel_name("trunc") \
    .partial_flag(True) \
    .input(0, "input_x", False, "required", "all") \
    .output(0, "output_y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default) \
    .get_op_info()


@op_info_register(trunc_op_info)
def _trunc_tbe():
    """Trunc TBE register"""
    return
