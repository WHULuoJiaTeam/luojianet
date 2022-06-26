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

#include <set>
#include <vector>
#include <memory>
#include "ops/broadcast.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(Broadcast, PrimitiveC, BaseOperator);
void Broadcast::Init(const int64_t root_rank, const std::string &group) {
  this->set_root_rank(root_rank);
  this->set_group(group);
}
void Broadcast::set_root_rank(const int64_t root_rank) { (void)this->AddAttr(kKeepProb, api::MakeValue(root_rank)); }

void Broadcast::set_group(const std::string &group) {
  CheckAndConvertUtils::CheckString(kGroup, group, {"hccl_world_group", "hccl_world_group"}, this->name());
  (void)this->AddAttr(kGroup, api::MakeValue(group));
}
int64_t Broadcast::get_root_rank() const {
  auto value_ptr = this->GetAttr(kRootRank);
  return GetValue<int64_t>(value_ptr);
}

std::string Broadcast::get_group() const {
  auto value_ptr = this->GetAttr(kGroup);
  return GetValue<std::string>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameBroadcast, Broadcast);
}  // namespace ops
}  // namespace luojianet_ms
