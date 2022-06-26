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

#ifndef LUOJIANET_MS_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_TYPES_H_
#define LUOJIANET_MS_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_TYPES_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "utils/hash_map.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/tensor.h"

#include "graph/tensor.h"
#include "external/ge/ge_api.h"

using GeTensor = ge::Tensor;

namespace luojianet_ms {
namespace transform {
enum Status : int { SUCCESS = 0, FAILED, INVALID_ARGUMENT, ALREADY_EXISTS, NOT_FOUND };

using MeTensor = luojianet_ms::tensor::Tensor;
using MeTensorPtr = std::shared_ptr<MeTensor>;
using MeDataType = luojianet_ms::TypeId;
using GeDataType = ge::DataType;
using GeFormat = ge::Format;
using GeShape = ge::Shape;
using GeTensorPtr = std::shared_ptr<GeTensor>;
using GeTensorDesc = ge::TensorDesc;
using AnfGraph = FuncGraph;
using AnfGraphPtr = FuncGraphPtr;
using Operator = ge::Operator;
using OperatorPtr = std::shared_ptr<ge::Operator>;
using DfGraph = ge::Graph;
using DfGraphPtr = std::shared_ptr<DfGraph>;
using TensorMap = luojianet_ms::HashMap<std::string, std::shared_ptr<MeTensor>>;

struct OutHandler {
  OperatorPtr op;
  std::string out;
  AnfNodePtr node;
  OutHandler() : op(nullptr), out(""), node(nullptr) {}
  OutHandler(const OperatorPtr &op, const std::string out, const AnfNodePtr &node = nullptr)
      : op(op), out(out), node(node) {}
};

struct ControlEdge {
  OperatorPtr src_op;
  OperatorPtr dest_op;
};
}  // namespace transform
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_TYPES_H_
