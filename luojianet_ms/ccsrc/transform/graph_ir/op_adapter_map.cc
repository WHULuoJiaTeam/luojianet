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

#include "transform/graph_ir/op_adapter_map.h"
#include <memory>
#include "graph/operator.h"

namespace luojianet_ms {
namespace transform {
template <>
luojianet_ms::HashMap<std::string, luojianet_ms::HashMap<int, std::string>> OpAdapter<ge::Operator>::cus_input_map_{};
template <>
luojianet_ms::HashMap<std::string, luojianet_ms::HashMap<int, std::string>> OpAdapter<ge::Operator>::cus_output_map_{};

luojianet_ms::HashMap<std::string, OpAdapterDescPtr> OpAdapterMap::adpt_map_ = {
  {kNameCustomOp, std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<Operator>>())}};

luojianet_ms::HashMap<std::string, OpAdapterDescPtr> &OpAdapterMap::get() { return adpt_map_; }
}  // namespace transform
}  // namespace luojianet_ms
