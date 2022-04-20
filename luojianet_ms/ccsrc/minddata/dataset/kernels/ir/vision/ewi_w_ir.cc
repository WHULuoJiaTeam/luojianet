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
#include "minddata/dataset/kernels/ir/vision/ewi_w_ir.h"

#include "minddata/dataset/kernels/image/ewi_w_op.h"

namespace luojianet_ms {
namespace dataset {

namespace vision {

EWI_WOperation::EWI_WOperation(float m, float n)
    :m_(m), 
     n_(n){}

// EWI_WOperation
EWI_WOperation::~EWI_WOperation() = default;

std::string EWI_WOperation::Name() const { return kEWI_WOperation; }

Status EWI_WOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> EWI_WOperation::Build() { 
        std::shared_ptr<EWI_WOp> tensor_op = std::make_shared<EWI_WOp>(m_, n_);
        return tensor_op; }

Status EWI_WOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["m"] = m_;
  args["n"] = n_;
  *out_json = args;
  return Status::OK();
}

}  // namespace vision
}  // namespace dataset
}  // namespace luojianet_ms
