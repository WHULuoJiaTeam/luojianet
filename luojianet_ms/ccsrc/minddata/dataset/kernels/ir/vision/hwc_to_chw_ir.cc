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

#include "minddata/dataset/kernels/ir/vision/hwc_to_chw_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/hwc_to_chw_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace luojianet_ms {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// HwcToChwOperation
HwcToChwOperation::~HwcToChwOperation() = default;

std::string HwcToChwOperation::Name() const { return kHwcToChwOperation; }

Status HwcToChwOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> HwcToChwOperation::Build() { return std::make_shared<HwcToChwOp>(); }

Status HwcToChwOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  *operation = std::make_shared<vision::HwcToChwOperation>();
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace luojianet_ms
