/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/random_resized_crop_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
RandomResizedCropOperation::RandomResizedCropOperation(const RandomResizedCropOperation &) = default;

// RandomResizedCropOperation
RandomResizedCropOperation::RandomResizedCropOperation(const std::vector<int32_t> &size,
                                                       const std::vector<float> &scale, const std::vector<float> &ratio,
                                                       InterpolationMode interpolation, int32_t max_attempts)
    : TensorOperation(true),
      size_(size),
      scale_(scale),
      ratio_(ratio),
      interpolation_(interpolation),
      max_attempts_(max_attempts) {}

RandomResizedCropOperation::~RandomResizedCropOperation() = default;

std::string RandomResizedCropOperation::Name() const { return kRandomResizedCropOperation; }

Status RandomResizedCropOperation::ValidateParams() {
  // size
  RETURN_IF_NOT_OK(ValidateVectorSize(Name(), size_));
  // scale
  RETURN_IF_NOT_OK(ValidateVectorScale(Name(), scale_));
  // ratio
  RETURN_IF_NOT_OK(ValidateVectorRatio(Name(), ratio_));
  // max_attempts
  if (max_attempts_ < 1) {
    std::string err_msg =
      Name() + ": max_attempts must be greater than or equal to 1, got: " + std::to_string(max_attempts_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // interpolation
  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour &&
      interpolation_ != InterpolationMode::kCubic && interpolation_ != InterpolationMode::kArea &&
      interpolation_ != InterpolationMode::kCubicPil) {
    std::string err_msg = "RandomResizedCrop: Invalid InterpolationMode, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomResizedCropOperation::Build() {
  constexpr size_t dimension_zero = 0;
  constexpr size_t dimension_one = 1;
  constexpr size_t size_two = 2;

  int32_t height = size_[dimension_zero];
  int32_t width = size_[dimension_zero];
  // User specified the width value.
  if (size_.size() == size_two) {
    width = size_[dimension_one];
  }
  std::shared_ptr<RandomCropAndResizeOp> tensor_op = std::make_shared<RandomCropAndResizeOp>(
    height, width, scale_[dimension_zero], scale_[dimension_one], ratio_[dimension_zero], ratio_[dimension_one],
    interpolation_, max_attempts_);
  return tensor_op;
}

Status RandomResizedCropOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = size_;
  args["scale"] = scale_;
  args["ratio"] = ratio_;
  args["interpolation"] = interpolation_;
  args["max_attempts"] = max_attempts_;
  *out_json = args;
  return Status::OK();
}

Status RandomResizedCropOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kRandomResizedCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "scale", kRandomResizedCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "ratio", kRandomResizedCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "interpolation", kRandomResizedCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "max_attempts", kRandomResizedCropOperation));
  std::vector<int32_t> size = op_params["size"];
  std::vector<float> scale = op_params["scale"];
  std::vector<float> ratio = op_params["ratio"];
  InterpolationMode interpolation = static_cast<InterpolationMode>(op_params["interpolation"]);
  int32_t max_attempts = op_params["max_attempts"];
  *operation = std::make_shared<vision::RandomResizedCropOperation>(size, scale, ratio, interpolation, max_attempts);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
