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

#include "minddata/dataset/audio/kernels/contrast_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status ContrastOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input tensor dimension, it should be greater than 0.
  RETURN_IF_NOT_OK(ValidateLowRank("Contrast", input, kMinAudioDim, "<..., time>"));
  // check input type, it should be DE_FLOAT
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Contrast", input));

  if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return Contrast(input, output, static_cast<double>(enhancement_amount_));
  } else {
    std::shared_ptr<Tensor> temp;
    RETURN_IF_NOT_OK(TypeCast(input, &temp, DataType(DataType::DE_FLOAT32)));
    return Contrast(temp, output, static_cast<float>(enhancement_amount_));
  }
}

Status ContrastOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("Contrast", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
