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

#define USE_DEPRECATED_API
#include "tools/common/graph_util.h"
#include <algorithm>
#include <functional>
#include <ctime>
#include <utility>
#include <set>
#include "tools/common/meta_graph_utils.h"
#include "schema/inner/model_generated.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/quantizer/bitpacking.h"
#include "tools/common/node_util.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"
#include "ops/make_tuple.h"
#include "tools/converter/converter_context.h"

namespace mindspore {
namespace lite {
namespace {
const int kZeroPointGap = 128;
}  // namespace
int SetFuncGraphOutput(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &outputs) {
  if (graph == nullptr || outputs.empty()) {
    MS_LOG(DEBUG) << "Input graph is nullptr or outputs is empty";
    return RET_INPUT_PARAM_INVALID;
  }
  if (outputs.size() == 1) {
    graph->set_output(outputs.front(), false);
    return RET_OK;
  }
  auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
  if (make_tuple_prim_ptr == nullptr) {
    MS_LOG(DEBUG) << "new MakeTuple failed";
    return lite::RET_NULL_PTR;
  }
  auto make_tuple_prim_c = make_tuple_prim_ptr->GetPrim();
  MS_CHECK_TRUE_MSG(make_tuple_prim_c != nullptr, lite::RET_NULL_PTR, "make_tuple_prim_c is nullptr");
  auto make_tuple_cnode = graph->NewCNode(make_tuple_prim_c, outputs);
  if (make_tuple_cnode == nullptr) {
    MS_LOG(DEBUG) << "new cnode failed";
    return lite::RET_NULL_PTR;
  }
  make_tuple_cnode->set_fullname_with_scope("return tuple");
  graph->set_output(make_tuple_cnode, false);
  return RET_OK;
}

OpDefCopyer GetSimpleOpCopyer() {
  return [](const CNodeT &inCNode) -> std::unique_ptr<CNodeT> {
    std::unique_ptr<CNodeT> newCNode = std::make_unique<CNodeT>();
    if (newCNode == nullptr) {
      return nullptr;
    }

    newCNode->name = inCNode.name;
    newCNode->quantType = inCNode.quantType;
    newCNode->primitive = std::make_unique<schema::PrimitiveT>();
    newCNode->primitive->value.type = inCNode.primitive->value.type;
    return newCNode;
  };
}

STATUS AddTensor2Node(schema::MetaGraphT *graphT, uint32_t nodeIdx, std::unique_ptr<TensorT> tensor,
                      InsertPlace place) {
  if (nodeIdx >= graphT->nodes.size()) {
    MS_LOG(ERROR) << "nodeIdx out of range: " << nodeIdx;
    return RET_PARAM_INVALID;
  }
  graphT->allTensors.emplace_back(std::move(tensor));
  uint32_t newTensorIdx = graphT->allTensors.size() - 1;
  auto node = graphT->nodes.at(nodeIdx).get();
  MS_CHECK_TRUE_MSG(node != nullptr, RET_NULL_PTR, "node is nullptr");
  if (place == kBefore) {
    node->inputIndex.emplace_back(newTensorIdx);
  } else {
    node->outputIndex.emplace_back(newTensorIdx);
  }
  return RET_OK;
}

STATUS ReplaceTensorOfNode(schema::MetaGraphT *graphT, uint32_t nodeIdx, uint32_t inTensorIdx,
                           std::unique_ptr<TensorT> tensor) {
  MS_ASSERT(graphT != nullptr);
  if (nodeIdx >= graphT->nodes.size()) {
    MS_LOG(ERROR) << "nodeIdx out of range: " << nodeIdx;
    return RET_PARAM_INVALID;
  }
  auto node = graphT->nodes.at(nodeIdx).get();
  MS_CHECK_TRUE_MSG(node != nullptr, RET_NULL_PTR, "node is nullptr");
  if (inTensorIdx >= graphT->allTensors.size()) {
    MS_LOG(ERROR) << "inTensorIdx out of range: " << nodeIdx;
    return RET_PARAM_INVALID;
  }
  if (!IsContain(node->inputIndex, inTensorIdx)) {
    MS_LOG(ERROR) << "inTensorIdx(" << inTensorIdx << ") is not a inputIdx of node(" << nodeIdx << ")";
    return RET_PARAM_INVALID;
  }
  graphT->allTensors.at(inTensorIdx).swap(tensor);
  return RET_OK;
}

NodeIter InsertNode(schema::MetaGraphT *graphT, uint32_t existNodeIdx, InsertPlace place, size_t inoutIndex,
                    std::unique_ptr<CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                    const OpDefCopyer &opDefCopyer) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(errorCode != nullptr);
  if (existNodeIdx >= graphT->nodes.size()) {
    MS_LOG(ERROR) << "nodeIdx out of range: " << existNodeIdx;
    return graphT->nodes.end();
  }
  auto node_iter = graphT->nodes.begin() + existNodeIdx;
  MS_ASSERT(node_iter != graphT->nodes.begin());
  MS_ASSERT((*node_iter) != nullptr);
  return InsertNode(graphT, node_iter, place, inoutIndex, std::move(toAddNode), errorCode, insert_num);
}

NodeIter InsertNode(schema::MetaGraphT *graphT, NodeIter existNodeIter, InsertPlace place, size_t inoutIndexIdx,
                    std::unique_ptr<CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                    const OpDefCopyer &opDefCopyer) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(errorCode != nullptr);
  if (place == kBefore) {
    return InsertNodeBefore(graphT, existNodeIter, inoutIndexIdx, std::move(toAddNode), errorCode, insert_num,
                            opDefCopyer);
  } else if (place == kAfter) {
    return InsertNodeAfter(graphT, existNodeIter, inoutIndexIdx, std::move(toAddNode), errorCode, insert_num,
                           opDefCopyer);
  } else {
    MS_LOG(ERROR) << "Invalid InsertPlace : " << place;
    return graphT->nodes.end();
  }
}

NodeIter InsertNodeBefore(schema::MetaGraphT *graphT, NodeIter existNodeIter, size_t inputIndexIdx,
                          std::unique_ptr<CNodeT> toAddNodeIn, STATUS *errorCode, int *insert_num,
                          const OpDefCopyer &opDefCopyer) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(errorCode != nullptr);
  auto &existNode = *existNodeIter;
  MS_ASSERT(existNode != nullptr);
  MS_ASSERT(existNode->inputIndex.size() > inputIndexIdx);
  MS_ASSERT(toAddNodeIn != nullptr);
  auto preTensorIdx = existNode->inputIndex.at(inputIndexIdx);
  MS_ASSERT(graphT->allTensors.size() > preTensorIdx);

  auto preNodeIdxes = GetInputNodeIdx(*graphT, *(existNode), inputIndexIdx);
  size_t insert_node_num = preNodeIdxes.empty() ? 1 : preNodeIdxes.size();
  std::vector<std::unique_ptr<CNodeT>> toAddNodes;
  for (size_t i = 0; i < insert_node_num; ++i) {
    auto &preTensor = graphT->allTensors.at(preTensorIdx);
    MS_ASSERT(preTensor != nullptr);
    auto toAddTensor = CopyTensorDefT(preTensor);
    if (toAddTensor == nullptr) {
      *errorCode = RET_NULL_PTR;
      MS_LOG(ERROR) << "Copy Tensor failed";
      return graphT->nodes.end();
    }
    toAddTensor->nodeType = NodeType_CNode;
    toAddTensor->refCount = 0;
    toAddTensor->data.clear();
    MS_ASSERT(toAddNodeIn->primitive != nullptr);
    if (toAddNodeIn->primitive->value.type == schema::PrimitiveType_QuantDTypeCast) {
      auto prim = toAddNodeIn->primitive->value.AsQuantDTypeCast();
      MS_ASSERT(prim != nullptr);
      if (prim->src_t == TypeId::kNumberTypeUInt8) {
        if (preTensor->dataType == TypeId::kNumberTypeUInt8) {
          toAddTensor->quantParams.front()->zeroPoint -= kZeroPointGap;
        } else {
          preTensor->quantParams.front()->zeroPoint += kZeroPointGap;
        }
      } else if (prim->dst_t == TypeId::kNumberTypeUInt8) {
        if (preTensor->dataType == TypeId::kNumberTypeInt8) {
          toAddTensor->quantParams.front()->zeroPoint += kZeroPointGap;
        } else {
          preTensor->quantParams.front()->zeroPoint -= kZeroPointGap;
        }
      }
      preTensor->dataType = prim->src_t;
      toAddTensor->dataType = prim->dst_t;
    }
    graphT->allTensors.emplace_back(std::move(toAddTensor));
    size_t toAddTensorIdx = graphT->allTensors.size() - 1;
    auto toAddNode = opDefCopyer(*toAddNodeIn);
    if (toAddNode == nullptr) {
      MS_LOG(ERROR) << "copy toAddNodeIn failed";
      *errorCode = RET_NULL_PTR;
      return graphT->nodes.end();
    }
    if (!preNodeIdxes.empty()) {
      toAddNode->name = toAddNodeIn->name + "_" + std::to_string(i);
    }
    toAddNode->inputIndex.clear();
    toAddNode->inputIndex.push_back(preTensorIdx);
    toAddNode->outputIndex.clear();
    toAddNode->outputIndex.push_back(toAddTensorIdx);
    for (auto iter = existNode->inputIndex.begin(); iter != existNode->inputIndex.end(); iter++) {
      if (*iter == preTensorIdx) {
        *iter = toAddTensorIdx;
        break;
      }
    }
    toAddNodes.emplace_back(std::move(toAddNode));
  }
  for (auto &toAddNode : toAddNodes) {
    existNodeIter = graphT->nodes.insert(existNodeIter, std::move(toAddNode));
    existNodeIter++;
    *insert_num += 1;
  }
  *errorCode = RET_OK;
  return existNodeIter;
}

NodeIter InsertNodeAfter(schema::MetaGraphT *graphT, NodeIter existNodeIter, size_t outputIndexIdx,
                         std::unique_ptr<schema::CNodeT> toAddNodeIn, STATUS *errorCode, int *insert_num,
                         const OpDefCopyer &opDefCopyer) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(errorCode != nullptr);
  auto &existNode = *existNodeIter;
  MS_ASSERT(existNode != nullptr);
  MS_ASSERT(existNode->outputIndex.size() > outputIndexIdx);
  MS_ASSERT(toAddNodeIn != nullptr);
  auto postTensorIdx = existNode->outputIndex.at(outputIndexIdx);
  MS_ASSERT(graphT->allTensors.size() > postTensorIdx);
  auto postNodeIdxes = GetOutputNodeIdx(*graphT, *(existNode), outputIndexIdx);
  bool is_output_index = IsContain(graphT->outputIndex, postTensorIdx);
  size_t insert_node_num = (postNodeIdxes.empty() || is_output_index) ? postNodeIdxes.size() + 1 : postNodeIdxes.size();
  bool has_insert_for_graph_out = postNodeIdxes.empty() || is_output_index;
  std::vector<std::unique_ptr<schema::CNodeT>> toAddNodes;
  for (size_t i = 0; i < insert_node_num; ++i) {
    auto &postTensor = graphT->allTensors.at(postTensorIdx);
    MS_ASSERT(postTensor != nullptr);
    auto toAddTensor = CopyTensorDefT(postTensor);
    if (toAddTensor == nullptr) {
      MS_LOG(ERROR) << "Copy TensorT failed";
      *errorCode = RET_NULL_PTR;
      return graphT->nodes.end();
    }
    toAddTensor->nodeType = NodeType_CNode;
    MS_ASSERT(toAddNodeIn->primitive != nullptr);
    if (toAddNodeIn->primitive->value.type == schema::PrimitiveType_QuantDTypeCast) {
      auto prim = toAddNodeIn->primitive->value.AsQuantDTypeCast();
      MS_ASSERT(prim != nullptr);
      if (prim->dst_t == TypeId::kNumberTypeUInt8) {
        if (postTensor->dataType == TypeId::kNumberTypeUInt8) {
          postTensor->quantParams.front()->zeroPoint -= kZeroPointGap;
        } else {
          toAddTensor->quantParams.front()->zeroPoint += kZeroPointGap;
        }
      } else if (prim->src_t == TypeId::kNumberTypeUInt8) {
        if (postTensor->dataType == TypeId::kNumberTypeUInt8) {
          toAddTensor->quantParams.front()->zeroPoint -= kZeroPointGap;
        } else {
          postTensor->quantParams.front()->zeroPoint += kZeroPointGap;
        }
      }
      postTensor->dataType = prim->src_t;
      toAddTensor->dataType = prim->dst_t;
    }
    graphT->allTensors.emplace_back(std::move(toAddTensor));
    size_t toAddTensorIdx = graphT->allTensors.size() - 1;
    auto toAddNode = opDefCopyer(*toAddNodeIn);
    if (toAddNode == nullptr) {
      MS_LOG(ERROR) << "copy toAddNodeIn failed";
      *errorCode = RET_NULL_PTR;
      return graphT->nodes.end();
    }
    toAddNode->inputIndex.clear();
    toAddNode->inputIndex.push_back(postTensorIdx);
    toAddNode->outputIndex.clear();
    toAddNode->outputIndex.push_back(toAddTensorIdx);
    if (!postNodeIdxes.empty()) {
      toAddNode->name = toAddNodeIn->name + "_" + std::to_string(i);
    }
    if (has_insert_for_graph_out) {
      ReplaceOutput(postTensorIdx, toAddTensorIdx, graphT);
      has_insert_for_graph_out = false;
    } else {
      auto &postNode = graphT->nodes.at(postNodeIdxes[is_output_index ? i - 1 : i]);
      for (auto iter = postNode->inputIndex.begin(); iter != postNode->inputIndex.end(); iter++) {
        if (*iter == postTensorIdx) {
          *iter = toAddTensorIdx;
        }
      }
    }
    toAddNodes.emplace_back(std::move(toAddNode));
  }
  for (auto &toAddNode : toAddNodes) {
    existNodeIter = graphT->nodes.insert(existNodeIter, std::move(toAddNode));
    existNodeIter++;
    *insert_num += 1;
  }
  *errorCode = RET_OK;
  return existNodeIter;
}

STATUS ValidateFileStr(const std::string &modelFile, const std::string &fileType) {
  if (modelFile.size() > fileType.size() && modelFile.substr(modelFile.size() - fileType.size()) == fileType) {
    return RET_OK;
  } else {
    return RET_ERROR;
  }
}

void SetSubgraphTensorIndices(schema::MetaGraphT *meta_graphT) {
  for (auto &subgraph : meta_graphT->subGraph) {
    std::vector<uint32_t> subgraph_indices{};
    subgraph_indices.assign(subgraph->inputIndices.begin(), subgraph->inputIndices.end());
    subgraph_indices.assign(subgraph->outputIndices.begin(), subgraph->outputIndices.end());
    for (auto &node_idx : subgraph->nodeIndices) {
      auto &node = meta_graphT->nodes.at(node_idx);
      for (auto &input_idx : node->inputIndex) {
        if (IsContain(subgraph_indices, input_idx)) {
          continue;
        } else {
          subgraph_indices.push_back(input_idx);
        }
      }
      for (auto &output_idx : node->outputIndex) {
        if (IsContain(subgraph_indices, output_idx)) {
          continue;
        } else {
          subgraph_indices.push_back(output_idx);
        }
      }
    }
    subgraph->tensorIndices.assign(subgraph_indices.begin(), subgraph_indices.end());
  }
}

std::string GetModelName(const std::string &modelFile) {
  std::string modelName = modelFile;
  modelName = modelName.substr(modelName.find_last_of('/') + 1);
  modelName = modelName.substr(0, modelName.find_last_of('.'));
  return modelName;
}

std::vector<int> GetTransposePerm(MetaGraphT *graph, const std::unique_ptr<CNodeT> &cnode) {
  MS_ASSERT(graph != nullptr && cnode != nullptr);
  std::vector<int> perm;
  if (cnode->primitive->value.type != schema::PrimitiveType_Transpose) {
    return perm;
  }
  if (cnode->inputIndex.size() < 2) {
    MS_LOG(ERROR) << "transpose node input size is less than 2.";
    return perm;
  }
  MS_ASSERT(cnode->outputIndex.at(1) < graph->allTensors.size());
  auto &perm_tensor = graph->allTensors.at(cnode->inputIndex.at(1));
  if (perm_tensor->data.empty()) {
    return perm;
  }
  MS_ASSERT(perm_tensor->dims.size() != 0);
  perm.resize(perm_tensor->dims[0]);
  if (memcpy_s(perm.data(), perm_tensor->dims[0] * sizeof(int), perm_tensor->data.data(),
               perm_tensor->dims[0] * sizeof(int)) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return {};
  }
  return perm;
}

TypeId GetAbstractTensorDtype(const abstract::AbstractTensorPtr &tensor) {
  if (tensor == nullptr || tensor->element() == nullptr) {
    MS_LOG(ERROR) << "abstract_tensor or abstract_tensor->element() is nullptr";
    return kTypeUnknown;
  }
  auto type_ptr = tensor->element()->GetTypeTrack();
  return type_ptr->type_id();
}

TypeId GetParameterDtype(const ParameterPtr &param_node) {
  auto abstract_base = param_node->abstract();
  auto abstract_tensor = abstract_base->cast<abstract::AbstractTensorPtr>();
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, kTypeUnknown, "Cast to abstract tensor failed!");
  auto type_ptr = abstract_tensor->element()->GetTypeTrack();
  return type_ptr->type_id();
}

STATUS UpdateFuncGraphInputsAndOutputsDtype(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  // update graph inputs dtype
  size_t idx = 0;
  for (auto &input : func_graph->get_inputs()) {
    TypeId type = GetParameterDtype(input->cast<ParameterPtr>());
    ConverterInnerContext::GetInstance()->UpdateGraphInputDType(idx, type);
    idx++;
  }
  // update graph outputs dtype
  auto graph_return = func_graph->get_return();
  idx = 0;
  for (auto &input : graph_return->inputs()) {
    if (input->isa<CNode>()) {
      if (utils::isa<abstract::AbstractTuple>(input->abstract())) {
        auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(input->abstract());
        if (tuple == nullptr) {
          MS_LOG(ERROR) << "tuple is nullptr";
          return RET_ERROR;
        }
        for (const auto &tuple_item : tuple->elements()) {
          if (utils::isa<abstract::AbstractTuple>(tuple_item)) {
            continue;
          }
          TypeId type = GetAbstractTensorDtype(tuple_item->cast<abstract::AbstractTensorPtr>());
          ConverterInnerContext::GetInstance()->UpdateGraphOutputDType(idx, type);
          idx++;
        }
      } else if (utils::isa<abstract::AbstractTensor>(input->abstract())) {
        TypeId type = GetAbstractTensorDtype(input->abstract()->cast<abstract::AbstractTensorPtr>());
        ConverterInnerContext::GetInstance()->UpdateGraphOutputDType(idx, type);
        idx++;
      } else {
        ConverterInnerContext::GetInstance()->UpdateGraphOutputDType(idx, kTypeUnknown);
        idx++;
      }
    }
  }
  return RET_OK;
}

STATUS UpdateGraphOutputName(schema::MetaGraphT *meta_graph) {
  auto output_names = ConverterInnerContext::GetInstance()->GetGraphOutputTensorNames();
  if (output_names.size() > meta_graph->outputIndex.size()) {
    MS_LOG(ERROR) << "the num of setting output_names is greater than actual, " << output_names.size() << " > "
                  << meta_graph->outputIndex.size() << ".";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  for (size_t idx = 0; idx < output_names.size(); idx++) {
    auto &tensor = meta_graph->allTensors.at(meta_graph->outputIndex.at(idx));
    tensor->name = output_names.at(idx);
  }
  return RET_OK;
}

int TransferMetaGraph(const schema::MetaGraphT &graph, void **model_buf, size_t *size) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "input model_buf invalid";
    return RET_ERROR;
  }
  if (size == nullptr) {
    MS_LOG(ERROR) << "input size invalid";
    return RET_ERROR;
  }

  /* model_buf malloc here, free outside */
  if (*model_buf != nullptr) {
    MS_LOG(ERROR) << "input model_buf must be nullptr";
    return RET_ERROR;
  }
  flatbuffers::FlatBufferBuilder builder(MAX_GRAPH_SIZE);
  auto offset = schema::MetaGraph::Pack(builder, &graph);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  *size = builder.GetSize();
  auto content = builder.GetBufferPointer();
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer nullptr";
    return RET_ERROR;
  }
  *model_buf = new char[*size];
  if (*model_buf == nullptr) {
    MS_LOG(ERROR) << "malloc model_buf failed";
    return RET_ERROR;
  }
  (void)memcpy(*model_buf, content, *size);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
