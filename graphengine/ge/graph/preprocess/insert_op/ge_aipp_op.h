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

#ifndef GE_GRAPH_PREPROCESS_INSERT_OP_GE_AIPP_OP_H_
#define GE_GRAPH_PREPROCESS_INSERT_OP_GE_AIPP_OP_H_

#include <utility>
#include <vector>
#include "framework/common/op/attr_value_util.h"
#include "graph/preprocess/insert_op/base_insert_op.h"
#include "proto/insert_op.pb.h"

namespace ge {
class AippOp : public InsertOpBase {
 public:
  AippOp() {}
  Status Init(domi::AippOpParams *aipp_params);

  ~AippOp() override;

  ///
  /// @ingroup domi_omg
  /// @brief Set Default Params
  ///
  Status SetDefaultParams() override;

  ///
  /// @ingroup domi_omg
  /// @brief Validate Params
  ///
  Status ValidateParams() override;

 protected:

  ///
  /// @ingroup domi_omg
  /// @brief Generate Op Desc
  ///
  Status GenerateOpDesc(ge::OpDescPtr op_desc) override;

  ///
  /// @ingroup domi_omg
  /// @brief Get Target Position
  /// @param [in] graph graph
  /// @param [in|out] target_input target input
  /// @param [in|out] target_edges target edges
  ///
  Status GetTargetPosition(ge::ComputeGraphPtr graph, ge::NodePtr &target_input,
                           std::vector<std::pair<ge::OutDataAnchorPtr, ge::InDataAnchorPtr>> &target_edges) override;

  Status InsertAippToGraph(ge::ComputeGraphPtr &graph,
                           std::string &aippConfigPath,
                           const uint32_t index) override ;

  domi::AippOpParams::AippMode GetAippMode() override;

 private:
  AippOp& operator=(const AippOp& aipp_op);
  AippOp(const AippOp& aipp_op);

  void ConvertParamToAttr(ge::GeAttrValue::NAMED_ATTRS &aipp_attrs);
  void SetCscDefaultValue();
  void SetDtcDefaultValue();
  NodePtr FindDataByIndex(const ComputeGraphPtr &graph, int rank);
  Status GetAndCheckTarget(const ComputeGraphPtr &graph, int rank, NodePtr &target, std::set<uint32_t> &edge_indexes);
  Status GetStaticTargetNode(const ComputeGraphPtr &graph, NodePtr &data_node, NodePtr &target);
  NodePtr CreateAipp(const OutDataAnchorPtr &out_anchor, const std::string &aippConfigPath, const uint32_t &index);
  Status CreateAippData(const NodePtr &aipp);
  Status AddNodeToGraph(const NodePtr &aipp_node, int64_t max_dynamic_aipp_size);
  Status AddAippAttrbutes(const OpDescPtr &op_desc, const std::string &aipp_cfg_path, const uint32_t &index);
  Status AddAttrToAippData(const OpDescPtr &aipp_data_op_desc);
  Status ConvertRelatedInputNameToRank();

  domi::AippOpParams *aipp_params_ = nullptr;
  ge::NodePtr aipp_node_ = nullptr;
  ge::NodePtr data_node_linked_aipp = nullptr;
};
}  // namespace ge

#endif  // GE_GRAPH_PREPROCESS_INSERT_OP_GE_AIPP_OP_H_

