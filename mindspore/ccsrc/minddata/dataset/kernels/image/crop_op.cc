/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/crop_op.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

Status CropOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImageRank("Crop", input->shape().Size()));
  int32_t input_h = static_cast<int>(input->shape()[0]);
  int32_t input_w = static_cast<int>(input->shape()[1]);
  CHECK_FAIL_RETURN_UNEXPECTED(y_ + height_ <= input_h, "Crop: Crop height dimension: " + std::to_string(y_ + height_) +
                                                          " exceeds image height: " + std::to_string(input_h));
  CHECK_FAIL_RETURN_UNEXPECTED(x_ + width_ <= input_w, "Crop: Crop width dimension: " + std::to_string(x_ + width_) +
                                                         " exceeds image width: " + std::to_string(input_w));
  return Crop(input, output, x_, y_, width_, height_);
}

Status CropOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out = TensorShape{height_, width_};
  if (inputs[0].Rank() == 2) {
    (void)outputs.emplace_back(out);
  }
  if (inputs[0].Rank() == 3) {
    (void)outputs.emplace_back(out.AppendDim(inputs[0][2]));
  }
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError,
                "Crop: invalid input shape, expected 2D or 3D input, but got input dimension is:" +
                  std::to_string(inputs[0].Rank()));
}
}  // namespace dataset
}  // namespace mindspore
