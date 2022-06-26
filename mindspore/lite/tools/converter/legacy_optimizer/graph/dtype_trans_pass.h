/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PREDICT_DTYPE_TRANS_PASS_H
#define MINDSPORE_PREDICT_DTYPE_TRANS_PASS_H

#include <memory>
#include <utility>
#include "tools/converter/optimizer.h"
#include "tools/common/graph_util.h"
#include "tools/converter/converter_flags.h"
#include "tools/common/tensor_util.h"

namespace mindspore {
namespace lite {
class DTypeTransPass : public GraphPass {
 public:
  DTypeTransPass(TypeId model_input_data_type, TypeId model_output_data_type)
      : id_(0), input_data_dtype(model_input_data_type), output_data_dtype(model_output_data_type) {}

  ~DTypeTransPass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;

 private:
  STATUS DoModelInputDTypeTrans(schema::MetaGraphT *graph);

  STATUS DoModelOutputDTypeTrans(schema::MetaGraphT *graph);

  STATUS DoNodeInoutDTypeTrans(schema::MetaGraphT *graph);

  STATUS InsetDTypeTransNodeForWrongDtypeQuantOp(schema::MetaGraphT *graph, NodeIter *iter);

  STATUS InsetDTypeTransNodeForUnsupportedInt8Op(schema::MetaGraphT *graph, NodeIter *iter);

  NodeIter InsertDTypeTransNode(schema::MetaGraphT *graph, NodeIter exist_node_iter, InsertPlace place,
                                size_t inout_idx, int32_t input_data_type, int32_t output_data_type,
                                STATUS *error_code);

  size_t id_;
  TypeId input_data_dtype = TypeId::kNumberTypeFloat;
  TypeId output_data_dtype = TypeId::kNumberTypeFloat;

  OpDefCopyer castOpCopyer = [](const schema::CNodeT &inCNode) -> std::unique_ptr<schema::CNodeT> {
    std::unique_ptr<schema::CNodeT> newCNode(new (std::nothrow) schema::CNodeT);
    if (newCNode == nullptr) {
      MS_LOG(ERROR) << "new CNodeT failed";
      return nullptr;
    }
    newCNode->name = inCNode.name;
    newCNode->quantType = inCNode.quantType;
    newCNode->primitive = std::make_unique<schema::PrimitiveT>();
    newCNode->primitive->value.type = inCNode.primitive->value.type;

    auto oldQuantDTypeCastParam = inCNode.primitive->value.AsQuantDTypeCast();
    auto QuantDTypeCastParam = new (std::nothrow) QuantDTypeCastT;
    if (QuantDTypeCastParam == nullptr) {
      MS_LOG(ERROR) << "new QuantDTypeCast failed";
      return nullptr;
    }
    QuantDTypeCastParam->src_t = oldQuantDTypeCastParam->src_t;
    QuantDTypeCastParam->dst_t = oldQuantDTypeCastParam->dst_t;
    newCNode->primitive->value.value = QuantDTypeCastParam;
    return newCNode;
  };
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_DTYPE_TRANS_PASS_H
