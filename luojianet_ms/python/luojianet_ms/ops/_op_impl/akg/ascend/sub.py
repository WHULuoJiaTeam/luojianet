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

"""Sub op"""
from luojianet_ms.ops.op_info_register import op_info_register, AkgAscendRegOp, DataType as DT

op_info = AkgAscendRegOp("Sub") \
    .fusion_type("ELEMWISE") \
    .input(0, "x") \
    .input(1, "y") \
    .output(0, "output") \
    .dtype_format(DT.F16_Default, DT.F16_Default, DT.F16_Default) \
    .dtype_format(DT.F32_Default, DT.F32_Default, DT.F32_Default) \
    .dtype_format(DT.I32_Default, DT.I32_Default, DT.I32_Default) \
    .dtype_format(DT.F16_5HD, DT.F16_5HD, DT.F16_5HD) \
    .dtype_format(DT.F32_5HD, DT.F32_5HD, DT.F32_5HD) \
    .dtype_format(DT.I32_5HD, DT.I32_5HD, DT.I32_5HD) \
    .dtype_format(DT.F16_FracZ, DT.F16_FracZ, DT.F16_FracZ) \
    .dtype_format(DT.F32_FracZ, DT.F32_FracZ, DT.F32_FracZ) \
    .dtype_format(DT.I32_FracZ, DT.I32_FracZ, DT.I32_FracZ) \
    .dtype_format(DT.F16_FracNZ, DT.F16_FracNZ, DT.F16_FracNZ) \
    .dtype_format(DT.F32_FracNZ, DT.F32_FracNZ, DT.F32_FracNZ) \
    .dtype_format(DT.I32_FracNZ, DT.I32_FracNZ, DT.I32_FracNZ) \
    .get_op_info()


@op_info_register(op_info)
def _sub_akg():
    """Sub Akg register"""
    return
