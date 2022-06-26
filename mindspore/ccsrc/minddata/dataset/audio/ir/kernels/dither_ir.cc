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

#include "minddata/dataset/audio/ir/kernels/dither_ir.h"

#include "minddata/dataset/audio/kernels/dither_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// DitherOperation
DitherOperation::DitherOperation(DensityFunction density_function, bool noise_shaping)
    : density_function_(density_function), noise_shaping_(noise_shaping) {
  random_op_ = true;
}

Status DitherOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DitherOperation::Build() {
  std::shared_ptr<DitherOp> tensor_op = std::make_shared<DitherOp>(density_function_, noise_shaping_);
  return tensor_op;
}

Status DitherOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["density_function"] = density_function_;
  args["noise_shaping"] = noise_shaping_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
