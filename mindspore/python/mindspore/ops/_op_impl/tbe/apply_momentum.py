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

"""ApplyMomentum op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

apply_momentum_op_info = TBERegOp("ApplyMomentum") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("apply_momentum.so") \
    .compute_cost(10) \
    .kernel_name("apply_momentum_d") \
    .partial_flag(True) \
    .attr("use_nesterov", "optional", "bool", "true,false", "false") \
    .input(0, "var", False, "required", "all") \
    .input(1, "accum", False, "required", "all") \
    .input(2, "lr", False, "required", "all") \
    .input(3, "grad", False, "required", "all") \
    .input(4, "momentum", False, "required", "all") \
    .output(0, "var", False, "required", "all") \
    .output(1, "accum", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_Default, DataType.F16_5HD,
                  DataType.F16_Default, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0, DataType.F16_Default, DataType.F16_C1HWNCoC0,
                  DataType.F16_Default, DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_Default, DataType.F16_FracZ,
                  DataType.F16_Default, DataType.F16_FracZ, DataType.F16_FracZ) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_Default, DataType.F32_5HD,
                  DataType.F32_Default, DataType.F32_5HD, DataType.F32_5HD) \
    .dtype_format(DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0, DataType.F32_Default, DataType.F32_C1HWNCoC0,
                  DataType.F32_Default, DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_Default, DataType.F32_FracZ,
                  DataType.F32_Default, DataType.F32_FracZ, DataType.F32_FracZ) \
    .dtype_format(DataType.F16_NDC1HWC0, DataType.F16_NDC1HWC0, DataType.F16_Default, DataType.F16_NDC1HWC0,
                  DataType.F16_Default, DataType.F16_NDC1HWC0, DataType.F16_NDC1HWC0) \
    .dtype_format(DataType.F16_FRACTAL_Z_3D, DataType.F16_FRACTAL_Z_3D, DataType.F16_Default, DataType.F16_FRACTAL_Z_3D,
                  DataType.F16_Default, DataType.F16_FRACTAL_Z_3D, DataType.F16_FRACTAL_Z_3D) \
    .dtype_format(DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0, DataType.F32_Default, DataType.F32_NDC1HWC0,
                  DataType.F32_Default, DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0) \
    .dtype_format(DataType.F32_FRACTAL_Z_3D, DataType.F32_FRACTAL_Z_3D, DataType.F32_Default, DataType.F32_FRACTAL_Z_3D,
                  DataType.F32_Default, DataType.F32_FRACTAL_Z_3D, DataType.F32_FRACTAL_Z_3D) \
    .get_op_info()


@op_info_register(apply_momentum_op_info)
def _apply_momentum_tbe():
    """ApplyMomentum TBE register"""
    return
