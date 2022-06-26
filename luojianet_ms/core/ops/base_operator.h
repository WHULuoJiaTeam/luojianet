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

#ifndef LUOJIANET_MS_CORE_OPS_BASE_OPERATOR_
#define LUOJIANET_MS_CORE_OPS_BASE_OPERATOR_

#include <memory>
#include <string>
#include <vector>

#include "mindapi/ir/primitive.h"

namespace luojianet_ms {
namespace abstract {
class AnalysisEngine;
using AnalysisEnginePtr = std::shared_ptr<AnalysisEngine>;

class AbstractBase;
using AbstractBasePtr = std::shared_ptr<AbstractBase>;
}  // namespace abstract
}  // namespace luojianet_ms

namespace luojianet_ms {
class Primitive;
using PrimitivePtr = std::shared_ptr<Primitive>;
}  // namespace luojianet_ms

namespace luojianet_ms {
namespace ops {
class PrimitiveC;
using PrimitiveCPtr = std::shared_ptr<PrimitiveC>;
class MIND_API BaseOperator : public api::Primitive {
 public:
  MIND_API_BASE_MEMBER(BaseOperator);
  explicit BaseOperator(const std::string &name);
  PrimitiveCPtr GetPrim();

 protected:
  void InitIOName(const std::vector<std::string> &inputs_name, const std::vector<std::string> &outputs_name);
};
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_BASE_OPERATOR_
