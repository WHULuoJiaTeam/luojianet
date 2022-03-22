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

#include "graph/manager/graph_context.h"

#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
GraphContext::GraphContext(const GraphNodePtr &graph_node) {
  if (graph_node == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "graphNode is NULL!");
    return;
  }

  compute_graph_ = graph_node->GetComputeGraph();
  current_graph_id_ = graph_node->GetGraphId();

  if (compute_graph_ == nullptr) {
    std::shared_ptr<const ge::Graph> graph = graph_node->GetGraph();
    if (graph == nullptr) {
      GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[Get][Graph] failed, compute_graph by graphNode is NULL!");
      return;
    }

    compute_graph_ = GraphUtils::GetComputeGraph(*graph);
    return;
  }
}

Status GraphContext::SetComputeGraph(const GraphNodePtr &graph_node) {
  if (graph_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param graph_node is nullptr, check invalid");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Check][Param] graphNode is NULL!");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  compute_graph_ = graph_node->GetComputeGraph();
  current_graph_id_ = graph_node->GetGraphId();

  if (compute_graph_ == nullptr) {
    std::shared_ptr<const ge::Graph> graph = graph_node->GetGraph();
    if (graph == nullptr) {
      REPORT_INNER_ERROR("E19999", "Param graph in graph_node is nullptr, check invalid");
      GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[Get][Graph] failed, compute_graph by graphNode is NULL!");
      return GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL;
    }

    compute_graph_ = GraphUtils::GetComputeGraph(*graph);
    return SUCCESS;
  }
  return SUCCESS;
}

Status GraphContext::Initialize(const std::map<std::string, std::string> &options) const { return SUCCESS; }

Status GraphContext::Finalize() const { return SUCCESS; }

Status GraphContext::GetVariableTensor(const std::string &var_data_name, GeTensor &returned_tensor) {
  if (var_data_name.empty()) {
    REPORT_INNER_ERROR("E19999", "Param var_data_name is empty, check invalid");
    GELOGE(GE_GRAPH_EMPTY_STRING_NAME, "[Check][Param] Variable data name is empty!");
    return GE_GRAPH_EMPTY_STRING_NAME;
  }

  if (GetVarNodeTensorTable().empty()) {
    REPORT_INNER_ERROR("E19999", "VarNodeTensorTable is empty, var_data_name:%s, check invalid",
                       var_data_name.c_str());
    GELOGE(GE_GRAPH_EMPTY_VARIABLE_TENSOR_TABLE, "[Check][Param] VarNodeTensorTable is empty, var_data_name:%s",
           var_data_name.c_str());
    return GE_GRAPH_EMPTY_VARIABLE_TENSOR_TABLE;
  }
  for (auto &var_record : GetVarNodeTensorTable()) {
    if (var_data_name == std::get<0>(var_record.first)) {
      returned_tensor.SetTensorDesc(var_record.second.GetTensorDesc());
      auto ret = returned_tensor.SetData(var_record.second.GetData());
      if (ret != SUCCESS) {
        REPORT_INNER_ERROR("E19999", "SetData to tensor fail, var_data_name:%s", var_data_name.c_str());
        GELOGE(ret, "[Set][Data] to Tensor failed, var_data_name:%s", var_data_name.c_str());
        return ret;
      }

      return SUCCESS;
    }
  }

  REPORT_INNER_ERROR("E19999", "VarRecord with data_name:%s does not exist, check invalid",
                     var_data_name.c_str());
  GELOGE(GE_GRAPH_VARIABLE_DOES_NOT_EXIST, "[Check][Param] VarRecord with data_name %s does NOT exist!",
         var_data_name.c_str());

  return GE_GRAPH_VARIABLE_DOES_NOT_EXIST;
}
}  // namespace ge
