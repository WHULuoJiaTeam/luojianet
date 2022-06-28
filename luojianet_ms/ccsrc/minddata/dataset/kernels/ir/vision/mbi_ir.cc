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
#include "minddata/dataset/kernels/ir/vision/mbi_ir.h"

#include "minddata/dataset/kernels/image/mbi_op.h"

namespace luojianet_ms {
namespace dataset {

namespace vision {

MBIOperation::MBIOperation( int32_t s_min,  int32_t s_max, int32_t delta_s)
    :s_min_(s_min), 
     s_max_(s_max), 
     delta_s_(delta_s){}

// MBIOperation
MBIOperation::~MBIOperation() = default;

std::string MBIOperation::Name() const { return kMBIOperation; }

Status MBIOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> MBIOperation::Build() { 
        std::shared_ptr<MBIOp> tensor_op = std::make_shared<MBIOp>(s_min_, s_max_, delta_s_);
        return tensor_op; }

Status MBIOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["s_min"] = s_min_;
  args["s_max"] = s_max_;
  args["delta_s"] = delta_s_;
  *out_json = args;
  return Status::OK();
}

}  // namespace vision
}  // namespace dataset
}  // namespace luojianet_ms
