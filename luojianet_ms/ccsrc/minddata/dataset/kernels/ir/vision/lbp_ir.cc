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
#include "minddata/dataset/kernels/ir/vision/lbp_ir.h"

#include "minddata/dataset/kernels/image/lbp_op.h"

namespace luojianet_ms {
namespace dataset {

namespace vision {

LBPOperation::LBPOperation(int N)
    :N_(N) {};

// LBPOperation
LBPOperation::~LBPOperation() = default;

std::string LBPOperation::Name() const { return kLBPOperation; }

Status LBPOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> LBPOperation::Build() { 
        std::shared_ptr<LBPOp> tensor_op = std::make_shared<LBPOp>(N_);
        return tensor_op; }

Status LBPOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["N"] = N_;
  *out_json = args;
  return Status::OK();
}

}  // namespace vision
}  // namespace dataset
}  // namespace luojianet_ms
