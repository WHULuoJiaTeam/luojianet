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

"""UniformCandidateSampler op"""
from luojianet_ms.ops.op_info_register import op_info_register, AiCPURegOp, DataType
uniform_candidate_sampler_op_info = AiCPURegOp("UniformCandidateSampler") \
    .fusion_type("OPAQUE") \
    .input(0, "true_classes", "required") \
    .output(0, "sampled_candidates", "required") \
    .output(1, "true_expected_count", "required") \
    .output(2, "true_expected_count", "required") \
    .attr("num_true", "int") \
    .attr("num_sampled", "int") \
    .attr("unique", "bool") \
    .attr("range_max", "int") \
    .attr("seed", "int") \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(uniform_candidate_sampler_op_info)
def _uniform_candidate_sampler_aicpu():
    """UniformCandidateSampler AiCPU register"""
    return
