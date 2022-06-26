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

"""Atan2 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

atan2_op_info = TBERegOp("Atan2") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("atan2.so") \
    .compute_cost(10) \
    .kernel_name("atan2") \
    .partial_flag(True) \
    .op_pattern("broadcast") \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_None, DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(atan2_op_info)
def _atan2_tbe():
    """Atan2 TBE register"""
    return
