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
#include "minddata/dataset/kernels/ir/vision/savi_ir.h"

#include "minddata/dataset/kernels/image/savi_op.h"

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace luojianet_ms {
namespace dataset {

namespace vision {

SAVIOperation::SAVIOperation(float L)
    :L_(L) {};

// SAVIOperation
SAVIOperation::~SAVIOperation() = default;

std::string SAVIOperation::Name() const { return kSAVIOperation; }

Status SAVIOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> SAVIOperation::Build() { 
        std::shared_ptr<SAVIOp> tensor_op = std::make_shared<SAVIOp>(L_);
        return tensor_op; }

Status SAVIOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["L"] = L_;
  *out_json = args;
  return Status::OK();
}

}  // namespace vision
}  // namespace dataset
}  // namespace luojianet_ms
