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

"""ConfusionMulGrad op"""
from luojianet_ms.ops.op_info_register import op_info_register, TBERegOp, DataType

confusion_mul_grad_op_info = TBERegOp("ConfusionMulGrad") \
    .fusion_type("OPAQUE") \
    .binfile_name("confusion_mul_grad.so") \
    .kernel_name("confusion_mul_grad") \
    .attr("axis", "required", "listInt", "all") \
    .attr("keep_dims", "required", "bool", "all") \
    .input(0, "input0", False, "required", "all") \
    .input(1, "input1", False, "required", "all") \
    .input(2, "input2", False, "required", "all") \
    .output(0, "output0", False, "required", "all") \
    .output(1, "output1", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(confusion_mul_grad_op_info)
def _confusion_mul_grad_tbe():
    """ConfusionMulGrad TBE register"""
    return
