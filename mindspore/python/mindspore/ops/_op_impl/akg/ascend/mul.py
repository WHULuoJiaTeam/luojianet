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

"""Mul op"""
from mindspore.ops.op_info_register import op_info_register, AkgAscendRegOp, DataType as DT

op_info = AkgAscendRegOp("Mul") \
    .fusion_type("ELEMWISE") \
    .input(0, "x") \
    .input(1, "y") \
    .output(0, "output") \
    .attr("x_shape", "required", "listInt") \
    .attr("y_shape", "required", "listInt") \
    .attr("data_format", "required", "listStr") \
    .dtype_format(DT.F16_Default, DT.F16_Default, DT.F16_Default) \
    .dtype_format(DT.F32_Default, DT.F32_Default, DT.F32_Default) \
    .dtype_format(DT.F16_5HD, DT.F16_5HD, DT.F16_5HD) \
    .dtype_format(DT.F32_5HD, DT.F32_5HD, DT.F32_5HD) \
    .dtype_format(DT.F16_FracZ, DT.F16_FracZ, DT.F16_FracZ) \
    .dtype_format(DT.F32_FracZ, DT.F32_FracZ, DT.F32_FracZ) \
    .dtype_format(DT.F16_FracNZ, DT.F16_FracNZ, DT.F16_FracNZ) \
    .dtype_format(DT.F32_FracNZ, DT.F32_FracNZ, DT.F32_FracNZ) \
    .get_op_info()


@op_info_register(op_info)
def _mul_akg():
    """Mul Akg register"""
    return
