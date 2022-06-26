# Copyright 2022 Huawei Technologies Co., Ltd
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

"""PriorityReplayBuffer op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType


prb_create_op_info = AiCPURegOp("PriorityReplayBufferCreate") \
    .fusion_type("OPAQUE") \
    .output(0, "handle", "required") \
    .attr("capacity", "int") \
    .attr("alpha", "float") \
    .attr("beta", "float") \
    .attr("schema", "listInt") \
    .attr("seed0", "int") \
    .attr("seed1", "int") \
    .dtype_format(DataType.I64_Default) \
    .get_op_info()


prb_push_op_info = AiCPURegOp("PriorityReplayBufferPush") \
    .input(0, "transition", "dynamic") \
    .output(0, "handle", "required") \
    .attr("handle", "int") \
    .dtype_format(DataType.BOOL_Default, DataType.I64_Default) \
    .dtype_format(DataType.I8_Default, DataType.I64_Default) \
    .dtype_format(DataType.I16_Default, DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F16_Default, DataType.I64_Default) \
    .dtype_format(DataType.U8_Default, DataType.I64_Default) \
    .dtype_format(DataType.U16_Default, DataType.I64_Default) \
    .dtype_format(DataType.U32_Default, DataType.I64_Default) \
    .dtype_format(DataType.U64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default) \
    .get_op_info()


prb_sample_op_info = AiCPURegOp("PriorityReplayBufferSample") \
    .output(0, "transitions", "dynamic") \
    .attr("handle", "int") \
    .attr("batch_size", "int") \
    .attr("schema", "listInt") \
    .dtype_format(DataType.BOOL_Default) \
    .dtype_format(DataType.I8_Default) \
    .dtype_format(DataType.I16_Default) \
    .dtype_format(DataType.I32_Default) \
    .dtype_format(DataType.I64_Default) \
    .dtype_format(DataType.F16_Default) \
    .dtype_format(DataType.U8_Default) \
    .dtype_format(DataType.U16_Default) \
    .dtype_format(DataType.U32_Default) \
    .dtype_format(DataType.U64_Default) \
    .dtype_format(DataType.F32_Default) \
    .get_op_info()


prb_update_op_info = AiCPURegOp("PriorityReplayBufferUpdate") \
    .input(0, "indices", "require") \
    .input(1, "priorities", "require") \
    .output(0, "handle", "require") \
    .attr("handle", "int") \
    .dtype_format(DataType.I64_Default, DataType.F32_Default, DataType.I64_Default) \
    .get_op_info()


@op_info_register(prb_create_op_info)
def _prb_create_op_cpu():
    """PriorityReplayBufferSample AICPU register"""
    return


@op_info_register(prb_push_op_info)
def _prb_push_op_cpu():
    """PriorityReplayBufferPush AICPU register"""
    return


@op_info_register(prb_sample_op_info)
def _prb_sample_op_cpu():
    """PriorityReplayBufferSample AICPU register"""
    return


@op_info_register(prb_update_op_info)
def _prb_update_op_cpu():
    """PriorityReplayBufferUpdate AICPU register"""
    return
