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

"""RandomLaplace op"""
from luojianet_ms.ops.op_info_register import op_info_register, AiCPURegOp, DataType

laplace_op_info = AiCPURegOp("StandardLaplace") \
    .fusion_type("OPAQUE") \
    .input(0, "shape", "required") \
    .input(1, "seed", "required") \
    .input(2, "seed2", "required") \
    .output(0, "output", "required") \
    .attr("seed", "int") \
    .attr("seed2", "int") \
    .dtype_format(DataType.I32_Default, DataType.I64_Default, DataType.I64_Default, DataType.F32_Default) \
    .get_op_info()

@op_info_register(laplace_op_info)
def _standard_laplace_aicpu():
    """RandomLaplace AiCPU register"""
    return
