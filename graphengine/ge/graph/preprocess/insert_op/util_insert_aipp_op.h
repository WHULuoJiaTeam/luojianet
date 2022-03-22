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

#ifndef GE_GRAPH_PREPROCESS_INSERT_OP_UTIL_INSERT_AIPP_OP_H_
#define GE_GRAPH_PREPROCESS_INSERT_OP_UTIL_INSERT_AIPP_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "graph/compute_graph.h"
#include "graph/preprocess/insert_op/base_insert_op.h"
#include "proto/insert_op.pb.h"

namespace ge {
enum AippType { OLD_TYPE, NEW_TYPE };

class InsertNewOpUtil {
 public:
  static InsertNewOpUtil &Instance() {
    thread_local InsertNewOpUtil instance;
    return instance;
  }

  Status Init();

  Status Parse(const char *conf_path);

  Status InsertAippOps(ge::ComputeGraphPtr &graph, std::string &aippConfigPath);

  void ClearNewOps();

  Status UpdateDataNodeByAipp(const ComputeGraphPtr &graph);

  Status RecordAIPPInfoToData(const ComputeGraphPtr &graph);

 private:
  Status CheckPositionNotRepeat();

  Status GetAippParams(const std::unique_ptr<domi::AippOpParams> &aippParams, const ge::NodePtr &aipp_node);

  Status CheckInputNamePositionNotRepeat();

  Status CheckInputRankPositionNoRepeat();

  Status CheckGraph(const ge::ComputeGraphPtr &graph);

  InsertNewOpUtil() = default;

  ~InsertNewOpUtil() = default;

  std::vector<std::unique_ptr<InsertOpBase>> insert_ops_;

  std::unique_ptr<domi::InsertNewOps> insert_op_conf_;

  void UpdateMultiBatchInputDims(const OpDescPtr &data_opdesc, Format &old_format);
  Status UpdatePrevNodeByAipp(NodePtr &node, std::set<NodePtr> &switchns);
  Status UpdateDataBySwitchN(const NodePtr &switchn, const NodePtr &data);
  Status GetDataRelatedNode(NodePtr &node, std::map<NodePtr, std::set<NodePtr>> &data_next_node_map);
  Status GetAllAipps(const NodePtr &data_node, const NodePtr &node, std::vector<NodePtr> &aipps);
  Status GetInputOutputInfo(NodePtr &data_node, NodePtr &aipp_node, std::string &input, std::string &output);
  Status SetModelInputDims(NodePtr &data_node, NodePtr &aipp_node);
  Status FindMaxSizeNode(const ComputeGraphPtr &graph, const NodePtr &case_node, map<uint32_t, int64_t> &max_sizes,
                         map<uint32_t, GeTensorDescPtr> &aipp_inputs);
  Status UpdateCaseNode(const ComputeGraphPtr &graph, const NodePtr &case_node);
};
}  // namespace ge

#endif  // GE_GRAPH_PREPROCESS_INSERT_OP_UTIL_INSERT_AIPP_OP_H_
