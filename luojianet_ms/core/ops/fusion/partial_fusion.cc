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

#include "ops/fusion/partial_fusion.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(PartialFusion, PrimitiveC, BaseOperator);
void PartialFusion::Init(const int64_t sub_graph_index) { this->set_sub_graph_index(sub_graph_index); }
void PartialFusion::set_sub_graph_index(const int64_t sub_graph_index) {
  (void)this->AddAttr(kSubGraphIndex, api::MakeValue(sub_graph_index));
}
int64_t PartialFusion::get_sub_graph_index() const {
  auto value_ptr = GetAttr(kSubGraphIndex);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNamePartialFusion, PartialFusion);
}  // namespace ops
}  // namespace luojianet_ms
