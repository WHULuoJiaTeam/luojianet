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

"""Unpack op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

unstack_op_info = TBERegOp("Unstack") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("unpack.so") \
    .compute_cost(10) \
    .kernel_name("unpack") \
    .partial_flag(True) \
    .attr("num", "optional", "int", "all", "None") \
    .attr("axis", "required", "int", "all") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "dynamic", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.None_None, DataType.None_None) \
    .get_op_info()


@op_info_register(unstack_op_info)
def _unpack_tbe():
    """Unpack TBE register"""
    return
