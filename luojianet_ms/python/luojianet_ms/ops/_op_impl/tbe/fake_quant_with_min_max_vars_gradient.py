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

"""FakeQuantWithMinMaxVars op"""
from luojianet_ms.ops.op_info_register import op_info_register, TBERegOp, DataType

fake_quant_with_min_max_vars_gradient_op_info = TBERegOp("FakeQuantWithMinMaxVarsGradient") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("fake_quant_with_min_max_vars_gradient.so") \
    .compute_cost(10) \
    .kernel_name("fake_quant_with_min_max_vars_gradient") \
    .partial_flag(True) \
    .attr("num_bits", "optional", "int", "all") \
    .attr("narrow_range", "optional", "bool", "all") \
    .input(0, "gradients", False, "required", "all") \
    .input(1, "x", False, "required", "all") \
    .input(2, "min", False, "required", "all") \
    .input(3, "max", False, "required", "all") \
    .output(0, "backprops_wrt_x", True, "required", "all") \
    .output(1, "backprops_wrt_min", True, "required", "all") \
    .output(2, "backprops_wrt_max", True, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(fake_quant_with_min_max_vars_gradient_op_info)
def _fake_quant_with_min_max_vars_gradient_tbe():
    """FakeQuantWithMinMaxVarsGradient TBE register"""
    return
