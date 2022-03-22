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

"""SoftmaxCrossEntropyWithLogits op"""
from luojianet_ms.ops.op_info_register import op_info_register, TBERegOp, DataType

softmax_cross_entropy_with_logits_op_info = TBERegOp("SoftmaxCrossEntropyWithLogits") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("softmax_cross_entropy_with_logits.so") \
    .compute_cost(10) \
    .kernel_name("softmax_cross_entropy_with_logits") \
    .partial_flag(True) \
    .input(0, "input_features", False, "required", "all") \
    .input(1, "input_labels", False, "required", "all") \
    .output(0, "output_loss", True, "required", "all") \
    .output(1, "output_backprop", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(softmax_cross_entropy_with_logits_op_info)
def _softmax_cross_entropy_with_logits_tbe():
    """SoftmaxCrossEntropyWithLogits TBE register"""
    return
