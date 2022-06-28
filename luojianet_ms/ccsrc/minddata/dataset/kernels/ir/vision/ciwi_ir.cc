/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/ir/vision/ciwi_ir.h"

#include "minddata/dataset/kernels/image/ciwi_op.h"

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace luojianet_ms {
namespace dataset {

namespace vision {

CIWIOperation::CIWIOperation(float digital_C)
    :digital_C_(digital_C) {}

// CIWIOperation
CIWIOperation::~CIWIOperation() = default;

std::string CIWIOperation::Name() const { return kCIWIOperation; }

Status CIWIOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> CIWIOperation::Build() { 
        std::shared_ptr<CIWIOp> tensor_op = std::make_shared<CIWIOp>(digital_C_);
        return tensor_op; }

Status CIWIOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["digital_C"] = digital_C_;
  *out_json = args;
  return Status::OK();
}

}  // namespace vision
}  // namespace dataset
}  // namespace luojianet_ms
