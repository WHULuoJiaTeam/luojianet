/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "coder/opcoders/nnacl/fp32/concat_fp32_coder.h"
#include <string>
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::lite::micro::nnacl {
int ConcatFP32Coder::Prepare(CoderContext *const context) {
  concat_param_ = reinterpret_cast<ConcatParameter *>(parameter_);
  return ReSize();
}

int ConcatFP32Coder::ReSize() {
  axis_ = concat_param_->axis_ >= 0 ? concat_param_->axis_
                                    : static_cast<int>(input_tensor_->shape().size()) + concat_param_->axis_;
  return RET_OK;
}

int ConcatFP32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/base/concat_base.h",
          },
          {
            "concat_base.c",
          });

  size_t input_num = input_tensors_.size();

  NNaclFp32Serializer code;
  code << "\t\tvoid *inputs_addr[] = {";
  for (size_t i = 0; i < input_num; ++i) {
    code << allocator_->GetRuntimeAddr(input_tensors_.at(i)) << ", ";
  }
  code << "};\n";

  size_t i;
  for (i = 0; i < input_num; ++i) {
    code << "\t\tint shape_" << i << "[] = {";
    for (auto &shape : input_tensors_.at(i)->shape()) {
      code << shape << ", ";
    }
    code << "};\n";
  }

  code << "\t\tint shape_" << i << "[] = {";
  for (auto &shape : output_tensor_->shape()) {
    code << shape << ", ";
  }
  code << "};\n";

  code << "\t\tint *inputs_output_shape[] = {";
  for (i = 0; i <= input_num; ++i) {
    code << "shape_" << i << ", ";
  }
  code << "};\n";

  code.CodeFunction("Concat", "inputs_addr", input_num, axis_, "inputs_output_shape", output_tensor_->shape().size(),
                    output_tensor_, 0, thread_num_, sizeof(float));
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Concat, CPUOpCoderCreator<ConcatFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
