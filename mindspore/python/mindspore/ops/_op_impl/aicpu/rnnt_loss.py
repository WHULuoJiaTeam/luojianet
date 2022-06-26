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

"""RNNTLoss op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

rnnt_loss_op_info = AiCPURegOp("RNNTLoss") \
    .fusion_type("OPAQUE") \
    .input(0, "acts", "required") \
    .input(1, "labels", "required") \
    .input(2, "input_lengths", "required") \
    .input(3, "label_lengths", "required") \
    .output(0, "costs", "required") \
    .output(1, "grads", "required") \
    .attr("blank_label", "int") \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()

@op_info_register(rnnt_loss_op_info)
def _rnnt_loss_aicpu():
    """RNNTLoss AiCPU register"""
    return
