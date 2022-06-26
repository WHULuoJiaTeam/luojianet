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

#include "minddata/dataset/audio/ir/kernels/vol_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/vol_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// Vol
VolOperation::VolOperation(float gain, GainType gain_type) : gain_(gain), gain_type_(gain_type) {}

VolOperation::~VolOperation() = default;

std::string VolOperation::Name() const { return kVolOperation; }

Status VolOperation::ValidateParams() {
  if (gain_type_ == GainType::kAmplitude) {
    RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Vol", "gain", gain_));
  }
  if (gain_type_ == GainType::kPower) {
    RETURN_IF_NOT_OK(ValidateFloatScalarPositive("Vol", "gain", gain_));
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> VolOperation::Build() {
  std::shared_ptr<VolOp> tensor_op = std::make_shared<VolOp>(gain_, gain_type_);
  return tensor_op;
}

Status VolOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["gain"] = gain_;
  args["gain_type"] = gain_type_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
