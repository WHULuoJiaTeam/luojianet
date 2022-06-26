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

"""FakeQuantWithMinMaxVars op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

fake_quant_with_min_max_vars_op_info = TBERegOp("FakeQuantWithMinMaxVars") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("fake_quant_with_min_max_vars.so") \
    .compute_cost(10) \
    .kernel_name("fake_quant_with_min_max_vars") \
    .partial_flag(True) \
    .attr("num_bits", "optional", "int", "all") \
    .attr("narrow_range", "optional", "bool", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "min", False, "required", "all") \
    .input(2, "max", False, "required", "all") \
    .output(0, "y", True, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(fake_quant_with_min_max_vars_op_info)
def _fake_quant_with_min_max_vars_tbe():
    """FakeQuantWithMinMaxVar TBE register"""
    return
