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
#include "minddata/dataset/engine/gnn/feature.h"

namespace luojianet_ms {
namespace dataset {
namespace gnn {

Feature::Feature(FeatureType type_name, std::shared_ptr<Tensor> value, bool is_shared_memory)
    : type_name_(type_name), value_(value), is_shared_memory_(is_shared_memory) {}

}  // namespace gnn
}  // namespace dataset
}  // namespace luojianet_ms
