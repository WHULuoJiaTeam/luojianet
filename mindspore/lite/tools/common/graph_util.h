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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_GRAPH_UTIL_H
#define MINDSPORE_LITE_TOOLS_COMMON_GRAPH_UTIL_H

#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <limits>
#include <functional>
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "src/common/graph_util.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
#define MAX_GRAPH_SIZE 1024
using STATUS = int;
enum InsertPlace { kBefore, kAfter };

using NodeIter = std::vector<std::unique_ptr<schema::CNodeT>>::iterator;

using OpDefCopyer = std::function<std::unique_ptr<schema::CNodeT>(const schema::CNodeT &)>;

OpDefCopyer GetSimpleOpCopyer();

int SetFuncGraphOutput(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &outputs);

STATUS AddTensor2Node(schema::MetaGraphT *graphT, uint32_t nodeIdx, std::unique_ptr<schema::TensorT> tensor,
                      InsertPlace place = kBefore);

STATUS ReplaceTensorOfNode(schema::MetaGraphT *graphT, uint32_t nodeIdx, uint32_t inTensorIdx,
                           std::unique_ptr<schema::TensorT> tensor);

NodeIter InsertNode(schema::MetaGraphT *graphT, uint32_t existNodeIdx, InsertPlace place, size_t inoutIndex,
                    std::unique_ptr<schema::CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                    const OpDefCopyer &opDefCopyer = GetSimpleOpCopyer());

NodeIter InsertNode(schema::MetaGraphT *graphT, NodeIter existNodeIter, InsertPlace place, size_t inoutIndexIdx,
                    std::unique_ptr<schema::CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                    const OpDefCopyer &opDefCopyer = GetSimpleOpCopyer());

NodeIter InsertNodeBefore(schema::MetaGraphT *graphT, NodeIter existNodeIter, size_t inputIndexIdx,
                          std::unique_ptr<schema::CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                          const OpDefCopyer &opDefCopyer);

NodeIter InsertNodeAfter(schema::MetaGraphT *graphT, NodeIter existNodeIter, size_t outputIndexIdx,
                         std::unique_ptr<schema::CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                         const OpDefCopyer &opDefCopyery);

STATUS ValidateFileStr(const std::string &modelFile, const std::string &fileType);

void SetSubgraphTensorIndices(schema::MetaGraphT *meta_graphT);

std::string GetModelName(const std::string &modelFile);

std::vector<int> GetTransposePerm(schema::MetaGraphT *graph, const std::unique_ptr<schema::CNodeT> &cnode);

TypeId GetAbstractTensorDtype(const abstract::AbstractTensorPtr &tensor);

TypeId GetParameterDtype(const ParameterPtr &param_node);

STATUS UpdateFuncGraphInputsAndOutputsDtype(const FuncGraphPtr &func_graph);

STATUS UpdateGraphOutputName(schema::MetaGraphT *meta_graph);

int TransferMetaGraph(const schema::MetaGraphT &graph, void **model_buf, size_t *size);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_COMMON_GRAPH_UTIL_H
