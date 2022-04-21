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
#include "minddata/dataset/kernels/ir/vision/mciwi_ir.h"

#include "minddata/dataset/kernels/image/mciwi_op.h"

namespace luojianet_ms {
namespace dataset {

namespace vision {

MCIWIOperation::MCIWIOperation() = default;

// MCIWIOperation
MCIWIOperation::~MCIWIOperation() = default;

std::string MCIWIOperation::Name() const { return kMCIWIOperation; }

Status MCIWIOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> MCIWIOperation::Build() { return std::make_shared<MCIWIOp>(); }

Status MCIWIOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  *operation = std::make_shared<vision::MCIWIOperation>();
  return Status::OK();
}

}  // namespace vision
}  // namespace dataset
}  // namespace luojianet_ms