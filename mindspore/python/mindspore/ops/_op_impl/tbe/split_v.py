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

"""SplitV op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

split_v_op_info = TBERegOp("SplitV") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("split_v_d.so") \
    .compute_cost(10) \
    .kernel_name("split_v_d") \
    .partial_flag(True) \
    .attr("size_splits", "required", "listInt", "all") \
    .attr("split_dim", "required", "int", "all") \
    .attr("num_split", "required", "int", "all") \
    .input(0, "input_x", False, "required", "all") \
    .output(0, "output", False, "dynamic", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.None_None, DataType.None_None) \
    .get_op_info()


@op_info_register(split_v_op_info)
def _split_v_tbe():
    """SplitV TBE register"""
    return
