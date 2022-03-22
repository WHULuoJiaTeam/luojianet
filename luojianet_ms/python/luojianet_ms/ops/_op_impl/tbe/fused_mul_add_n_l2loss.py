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

"""FusedMulAddNL2loss op"""
from luojianet_ms.ops.op_info_register import op_info_register, TBERegOp, DataType

fused_mul_add_n_l2loss_op_info = TBERegOp("FusedMulAddNL2loss") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("fused_mul_addn_l2_loss.so") \
    .compute_cost(10) \
    .kernel_name("fused_mul_addn_l2_loss") \
    .partial_flag(True) \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .input(2, "x3", False, "required", "all") \
    .output(0, "y1", False, "required", "all") \
    .output(1, "y2", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_Default,
                  DataType.F16_5HD, DataType.F16_Default) \
    .dtype_format(DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0, DataType.F16_Default,
                  DataType.F16_C1HWNCoC0, DataType.F16_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_Default,
                  DataType.F16_FracZ, DataType.F16_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_Default,
                  DataType.F32_5HD, DataType.F32_Default) \
    .dtype_format(DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0, DataType.F32_Default,
                  DataType.F32_C1HWNCoC0, DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_Default,
                  DataType.F32_FracZ, DataType.F32_Default) \
    .get_op_info()


@op_info_register(fused_mul_add_n_l2loss_op_info)
def _fused_mul_add_n_l2loss_tbe():
    """FusedMulAddNL2loss TBE register"""
    return
