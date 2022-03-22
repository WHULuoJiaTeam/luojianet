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

#include "graph/passes/inplace_support_check_pass.h"
#include "framework/common/debug/log.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
constexpr uint32_t kInplaceSupportOutputIndex = 0;
constexpr uint32_t kInplaceSupportOutputNum = 1;
const std::set<std::string> kSrcNodeTypes = { ge::DATA, ge::ANN_DATA, ge::AIPPDATA,
                                              ge::CONSTANT, ge::CONSTANTOP,
                                              ge::VARIABLE, ge::VARIABLEV2 };
}
Status InplaceSupportCheckPass::Run(NodePtr &node) {
  GELOGD("InplaceSupportCheckPass running");
  if (node->GetAllOutDataAnchorsSize() != kInplaceSupportOutputNum) {
    GELOGD("output num of node %s is not %u, skip InplaceSupportCheckPass",
           node->GetName().c_str(), kInplaceSupportOutputNum);
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(node->GetOpDesc());
  const DataType &output_type = node->GetOpDesc()->GetOutputDesc(kInplaceSupportOutputIndex).GetDataType();
  const GeShape &output_shape = node->GetOpDesc()->GetOutputDesc(kInplaceSupportOutputIndex).GetShape();
  GELOGD("process InplaceSupportCheckPass on node %s", node->GetName().c_str());
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    const auto &peer_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_data_anchor == nullptr) {
      continue;
    }
    auto in_node = peer_data_anchor->GetOwnerNode();
    if (kSrcNodeTypes.count(in_node->GetType()) > 0) {
      GELOGD("meet src_node %s", in_node->GetName().c_str());
      continue;
    }
    if (peer_data_anchor->GetPeerInDataNodesSize() != kInplaceSupportOutputNum) {
      GELOGD("peer_data_anchor links with multi in_data_anchors");
      continue;
    }

    int32_t inplace_input_idx = in_data_anchor->GetIdx();
    const DataType &input_type = node->GetOpDesc()->GetInputDesc(inplace_input_idx).GetDataType();
    const GeShape &input_shape = node->GetOpDesc()->GetInputDesc(inplace_input_idx).GetShape();
    if (input_type !=  output_type) {
      GELOGW("DataType mismatch, in_idx=%d, input_type=%u, output_type=%u", inplace_input_idx, input_type, output_type);
      continue;
    }
    if (input_shape.GetDims() != output_shape.GetDims()) {
      GELOGW("Shape mismatch, in_idx=%d, input_shape=[%s], output_shape=[%s]",
             inplace_input_idx, input_shape.ToString().c_str(), output_shape.ToString().c_str());
      continue;
    }

    GELOGD("add attr INPLACE_SUPPORT_INPUT_INDEX on node %s, input_idx=%d", node->GetName().c_str(), inplace_input_idx);
    if (!AttrUtils::SetInt(node->GetOpDesc()->MutableOutputDesc(kInplaceSupportOutputIndex),
                           INPLACE_SUPPORT_INPUT_INDEX, inplace_input_idx)) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to output:%u tensor of op:%s(%s) failed",
                        INPLACE_SUPPORT_INPUT_INDEX.c_str(), kInplaceSupportOutputIndex,
                        node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to output:%u tensor of op:%s(%s) failed",
             INPLACE_SUPPORT_INPUT_INDEX.c_str(), kInplaceSupportOutputIndex,
             node->GetName().c_str(), node->GetType().c_str());
      return FAILED;
    }
    AddRePassNode(node);
    break;
  }
  return SUCCESS;
}
}  // namespace ge
