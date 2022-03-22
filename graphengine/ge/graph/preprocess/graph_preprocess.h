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

#ifndef GE_GRAPH_PREPROCESS_GRAPH_PREPROCESS_H_
#define GE_GRAPH_PREPROCESS_GRAPH_PREPROCESS_H_
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "framework/common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/model_parser/model_parser.h"
#include "common/properties_manager.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/manager/util/variable_accelerate_ctrl.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "framework/omg/omg_inner_types.h"
#include "runtime/context.h"

namespace ge {
class GraphPrepare {
 public:
  GraphPrepare();
  virtual ~GraphPrepare();
  GraphPrepare(const GraphPrepare &in) = delete;
  GraphPrepare &operator=(const GraphPrepare &in) = delete;
  Status PrepareDynShape(const GraphNodePtr &graph_node,
                         const std::vector<GeTensor> &user_input,
                         ge::ComputeGraphPtr &compute_graph,
                         uint64_t session_id = 0);
  Status RecordAIPPInfo(ge::ComputeGraphPtr &compute_graph);
  Status PrepareRunningFormatRefiner();
  void SetOptions(const GraphManagerOptions &options);
  Status GenerateInfershapeGraph(ConstGraphPtr graph);
  Status SwitchOpOptimize(ComputeGraphPtr &compute_graph);

 private:
  Status Init(const ge::Graph &graph, uint64_t session_id = 0);
  Status CheckGraph();
  Status CheckRefInputNode(const NodePtr &node, const std::string &input_name,
                           const std::set<NodePtr> &ref_nodes);
  Status CheckRefOp();
  Status SetRtContext(rtContext_t rt_context, rtCtxMode_t mode);
  Status AdjustDataOpOutput(const NodePtr &node);
  Status CheckInternalFormat(const NodePtr &input_node, const GeTensorDesc &desc);
  Status UpdateDataInputOutputDesc(GeAttrValue::INT index, OpDescPtr &op, GeTensorDesc &desc);
  Status UpdateInput(const std::vector<GeTensor> &user_input, const std::map<string, string> &graph_option);
  Status CheckAndUpdateInput(const std::vector<GeTensor> &user_input, const std::map<string, string> &graph_option);
  Status CheckConstOp();
  Status VerifyConstOp(const NodePtr &node);
  Status CheckUserInput(const std::vector<GeTensor> &user_input);
  Status UpdateDataNetOutputByStorageFormat();
  Status PrepareOptimize();
  Status InferShapeForPreprocess();
  Status TryDoAipp();
  Status UpdateVariableFormats(ComputeGraphPtr &graph);
  Status FormatAndShapeProcess();
  Status ResourcePairProcess(const std::string &action);
  Status SaveOriginalGraphToOmModel();
  Status ProcessNetOutput();
  Status ProcessBeforeInfershape();
  Status UpdateInputOutputByOptions();
  Status CtrlFlowPreProcess();

  bool IsTansDataOpData(const ge::NodePtr &var_node);

  Status GraphEquivalentTransformation();
  void TypeConversionOfConstant();
  bool IsDynamicDims(const NodePtr &input_node);

  ge::ComputeGraphPtr compute_graph_;
  GraphManagerOptions options_;
  uint64_t session_id_ = 0;
};
}  // namespace ge
#endif  // GE_GRAPH_PREPROCESS_GRAPH_PREPROCESS_H_
