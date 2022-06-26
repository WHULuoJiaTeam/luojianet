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

"""CTCLoss op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
ctcloss_op_info = AiCPURegOp("CTCLoss") \
    .fusion_type("OPAQUE") \
    .input(0, "inputs", "required") \
    .input(1, "labels_indices", "required") \
    .input(2, "labels_values", "required") \
    .input(3, "sequence_length", "required") \
    .output(0, "loss", "required") \
    .output(1, "gradient", "required") \
    .attr("preprocess_collapse_repeated", "bool") \
    .attr("ctc_merge_repeated", "bool") \
    .attr("ignore_longer_outputs_than_inputs", "bool") \
    .dtype_format(DataType.F32_Default, DataType.I64_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.I64_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()

@op_info_register(ctcloss_op_info)
def _ctcloss_aicpu():
    """CTCLoss AiCPU register"""
    return
