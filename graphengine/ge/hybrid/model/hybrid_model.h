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

#ifndef GE_HYBRID_HYBRID_GRAPH_H_
#define GE_HYBRID_HYBRID_GRAPH_H_

#include <vector>
#include <queue>
#include <memory>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/load/model_manager/data_inputer.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/node.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/model/node_item.h"
#include "hybrid/model/graph_item.h"
#include "common/model/ge_root_model.h"

namespace ge {
namespace hybrid {
class HybridModel {
 public:
  explicit HybridModel(GeRootModelPtr ge_model);

  ~HybridModel();

  Status Init(bool is_single_op = false);

  const NodeItem *GetNodeItem(const NodePtr &node) const;

  uint64_t GetSessionId() const {
    return root_runtime_param_.session_id;
  }

  void *GetGlobalStep() const;

  GeModelPtr GetGeModel(const NodePtr &node) const;

  NodeItem *MutableNodeItem(const NodePtr &node);

  size_t TotalVarMemSize() const {
    return root_runtime_param_.var_size;
  }

  const uint8_t* GetVarMemBase() const {
    return var_mem_base_;
  }

  void SetDeviceId(uint32_t device_id)  {
    device_id_ = device_id;
  }

  uint32_t GetDeviceId() {
    return device_id_;
  }

  void SetModelId(uint32_t model_id) {
    model_id_ = model_id;
  }

  void SetOmName(const string &om_name) {
    om_name_ = om_name;
  }

  const std::string &GetOmName() const {
    return om_name_;
  }

  uint32_t GetModelId() const {
    return model_id_;
  }

  bool IsSingleOp() const {
    return is_single_op_;
  }

  TensorValue* GetVariable(const string &name) const;

  NodePtr GetVariableNode(const string &name) const;

  TensorValue* GetTensor(const NodePtr &node) const;

  TensorBuffer* GetModelWeight(const std::string &subgraph_name) const;

  const std::map<int64_t, std::vector<std::pair<int, Tensor>>> &GetHostTensors() const;

  const std::vector<domi::TaskDef>* GetTaskDefs(const NodePtr &node) const;

  const GraphItem *GetRootGraphItem() const;

  const ComputeGraphPtr &GetRootGraph() const;

  const GraphItem *GetSubgraphItem(const std::string &graph_name) const;

  const GraphItem *GetSubgraphItem(const ComputeGraphPtr &subgraph) const;

  const string &GetModelName() const;

  Status GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type);

  void GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order);

  void GetModelAttr(std::vector<std::string> &dynamic_output_shape_info);

  Status GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                vector<InputOutputDescInfo> &output_desc,
                                std::vector<uint32_t> &input_formats,
                                std::vector<uint32_t> &outputFormats);

  Status GetInputDescInfo(vector<InputOutputDescInfo> &input_desc, std::vector<uint32_t> &formats);

  void CreateOutput(ConstGeTensorDescPtr &output_desc, InputOutputDescInfo &output, uint32_t &format_result);

  Status GetOutputDescInfo(vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &formats);

  void CreateInputDimsInfo(const OpDescPtr &op_desc, InputOutputDescInfo &input);

  void SetModelDescVersion(bool is_new_model_desc) { is_new_model_desc_ = is_new_model_desc; }

  void SetInputDimsAndShapeRangesInfo(const vector<int64_t> &model_input_dims,
                                      std::vector<std::pair<int64_t, int64_t>> &shape_ranges,
                                      InputOutputDescInfo &input);
  void SaveSpecifyAttrValues();

  Status GetOpAttr(const std::string &op_name, const std::string &attr_name, std::string &attr_value) const;

 private:
  friend class HybridModelBuilder;
  friend class HybridModelAsyncExecutor;

  TensorValue* GetConstant(const NodePtr &node) const;

  std::string model_name_;
  GeRootModelPtr ge_root_model_;
  std::map<uint32_t, NodeItem *> input_nodes_;
  ComputeGraphPtr root_graph_;
  ComputeGraphPtr orig_root_graph_;
  std::map<std::string, NodePtr> device_variable_nodes_; //lint !e148
  std::map<std::string, NodePtr> host_variable_nodes_; //lint !e148
  std::map<std::string, std::unique_ptr<TensorValue>> variable_tensors_;
  std::map<NodePtr, std::unique_ptr<TensorValue>> constant_tensors_;
  std::map<NodePtr, std::vector<domi::TaskDef>> task_defs_;
  std::map<NodePtr, GeModelPtr> known_shape_sub_models_;

  std::unique_ptr<GraphItem> root_graph_item_;
  std::map<std::string, std::unique_ptr<GraphItem>> subgraph_items_;
  std::map<NodePtr, std::unique_ptr<NodeItem>> node_items_;
  std::map<int64_t, std::vector<std::pair<int, Tensor>>> host_tensors_;

  bool is_new_model_desc_ = false;    // support aipp
  bool is_single_op_ = false;

  // runtime fields
  uint32_t device_id_ = 0;
  uint32_t model_id_ = 0;
  uint8_t *var_mem_base_ = nullptr;
  std::map<string, std::unique_ptr<TensorBuffer>> weight_buffer_map_;
  RuntimeParam root_runtime_param_;
  string om_name_;
  std::unique_ptr<TensorBuffer> global_step_;
  // op name to attrs mapping
  std::map<std::string, std::map<std::string, std::vector<std::string>>> op_name_to_attrs_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_HYBRID_GRAPH_H_
