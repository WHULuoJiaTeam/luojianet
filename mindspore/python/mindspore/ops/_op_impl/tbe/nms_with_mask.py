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

"""NMSWithMask op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

nms_with_mask_op_info = TBERegOp("NMSWithMask") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("nms_with_mask.so") \
    .compute_cost(10) \
    .kernel_name("nms_with_mask") \
    .partial_flag(True) \
    .attr("iou_threshold", "optional", "float", "all") \
    .input(0, "box_scores", False, "required", "all") \
    .output(0, "selected_boxes", False, "required", "all") \
    .output(0, "selected_idx", False, "required", "all") \
    .output(0, "selected_mask", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.I32_Default, DataType.U8_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.I32_Default, DataType.BOOL_Default) \
    .get_op_info()


@op_info_register(nms_with_mask_op_info)
def _nms_with_mask_tbe():
    """NMSWithMask TBE register"""
    return
