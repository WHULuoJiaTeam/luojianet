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

#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_horizontal_flip_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomHorizontalFlipOperation
RandomHorizontalFlipOperation::RandomHorizontalFlipOperation(float prob) : TensorOperation(true), probability_(prob) {}

RandomHorizontalFlipOperation::~RandomHorizontalFlipOperation() = default;

std::string RandomHorizontalFlipOperation::Name() const { return kRandomHorizontalFlipOperation; }

Status RandomHorizontalFlipOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomHorizontalFlip", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomHorizontalFlipOperation::Build() {
  std::shared_ptr<RandomHorizontalFlipOp> tensor_op = std::make_shared<RandomHorizontalFlipOp>(probability_);
  return tensor_op;
}

Status RandomHorizontalFlipOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["prob"] = probability_;
  return Status::OK();
}

Status RandomHorizontalFlipOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomHorizontalFlipOperation));
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomHorizontalFlipOperation>(prob);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
