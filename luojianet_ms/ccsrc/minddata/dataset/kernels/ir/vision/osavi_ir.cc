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

#include "minddata/dataset/kernels/ir/vision/osavi_ir.h"

#include "minddata/dataset/kernels/image/osavi_op.h"

namespace luojianet_ms {
namespace dataset {

namespace vision {

OSAVIOperation::OSAVIOperation(float theta)
    :theta_(theta) {};

// OSAVIOperation
OSAVIOperation::~OSAVIOperation() = default;

std::string OSAVIOperation::Name() const { return kOSAVIOperation; }

Status OSAVIOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> OSAVIOperation::Build() { 
        std::shared_ptr<OSAVIOp> tensor_op = std::make_shared<OSAVIOp>(theta_);
        return tensor_op; }

Status OSAVIOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["theta"] = theta_;
  *out_json = args;
  return Status::OK();
}

}  // namespace vision
}  // namespace dataset
}  // namespace luojianet_ms
