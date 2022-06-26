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


"""MaxPool3D op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

max_pool3d_op_info = TBERegOp("MaxPool3D") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("max_pool3d.so") \
    .compute_cost(10) \
    .kernel_name("max_pool3d") \
    .partial_flag(True) \
    .attr("kernel_size", "required", "listInt", "all") \
    .attr("strides", "required", "listInt", "all") \
    .attr("pad_mode", "required", "str", "all") \
    .attr("pad_list", "optional", "listInt", "all", "0,0,0") \
    .attr("dilation", "optional", "listInt", "all", "1,1,1") \
    .attr("ceil_mode", "optional", "int", "all", "0") \
    .attr("format", "optional", "str", "all") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.None_None, DataType.None_None) \
    .get_op_info()


@op_info_register(max_pool3d_op_info)
def _max_pool_3d_tbe():
    """MaxPool3D TBE register"""
    return
