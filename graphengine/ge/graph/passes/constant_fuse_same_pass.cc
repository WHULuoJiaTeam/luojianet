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

#include "graph/passes/constant_fuse_same_pass.h"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
const size_t kCorrectNum = 1;
const char *const kOriginElementNumAttrName = "origin_element_num";

bool CheckConstInAndOut(const NodePtr &node) {
  // has none in control
  // has one out data anchor
  if ((node->GetInControlNodes().empty()) && (node->GetAllOutDataAnchorsSize() == kCorrectNum)) {
    return true;
  }
  return false;
}

void GetOutDataNodeToIndexMap(NodePtr &node, std::map<string, InDataAnchorPtr> &out_node_to_indexs) {
  auto out_data_anchor = node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL_JUST_RETURN(out_data_anchor);
  auto peer_in_anchors = out_data_anchor->GetPeerInDataAnchors();
  if (!peer_in_anchors.empty()) {
    for (auto &anchor : peer_in_anchors) {
      int index = anchor->GetIdx();
      NodePtr out_node = anchor->GetOwnerNode();
      if (out_node == nullptr) {
        continue;
      }
      string key_name = out_node->GetName() + "-" + std::to_string(index);
      out_node_to_indexs[key_name] = anchor;
    }
  }
}
}  // namespace

Status ConstantFuseSamePass::Run(ge::ComputeGraphPtr graph) {
  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Check][Param] Compute graph is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  GELOGI("ConstantFuseSamePass in.");

  std::map<SameConstKey, std::vector<NodePtr>> fuse_nodes;
  GetFuseConstNodes(graph, fuse_nodes);

  return FuseConstNodes(graph, fuse_nodes);
}

void ConstantFuseSamePass::GetFuseConstNodes(ComputeGraphPtr &graph,
                                             std::map<SameConstKey, std::vector<NodePtr>> &fuse_nodes) {
  int total_const_nums = 0;
  int insert_const_nums = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() != CONSTANT && node->GetType() != CONSTANTOP) {
      continue;
    }
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    ++total_const_nums;

    if (!CheckConstInAndOut(node)) {
      GELOGD("The const %s does not support to fusion, skip it", node->GetName().c_str());
      continue;
    }

    GeTensorPtr weight;
    if (!AttrUtils::MutableTensor(op_desc, ATTR_NAME_WEIGHTS, weight)) {
      GELOGW("The const node %s does not have weight attr, skip it", node->GetName().c_str());
      continue;
    }
    int64_t origin_element_num = -1;
    if (!AttrUtils::GetInt(weight->MutableTensorDesc(), kOriginElementNumAttrName, origin_element_num)) {
      GELOGI("The const %s does not have origin element num attribute, skip it", node->GetName().c_str());
      continue;
    }
    if (origin_element_num != 1) {
      GELOGI("The const %s origin element num %ld, does not support to fusion now", node->GetName().c_str(),
             origin_element_num);
      continue;
    }

    auto output_tensor = op_desc->MutableOutputDesc(0);
    if (output_tensor == nullptr) {
      GELOGW("The const %s does not have output 0, skip to fusion", node->GetName().c_str());
      continue;
    }
    auto data_type = output_tensor->GetDataType();
    auto type_size = GetSizeByDataType(data_type);
    if (type_size < 0) {
      GELOGI("The data type of const %s does not support fusion, data type %s", node->GetName().c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str());
      continue;
    }
    if ((type_size != 0) && (weight->MutableData().GetAlignedPtr() == nullptr)) {
      GELOGW("aligned_ptr is null while size is not 0");
      continue;
    }
    ++insert_const_nums;

    SameConstKey map_key;
    map_key.data_size = type_size;
    map_key.aligned_ptr = weight->MutableData().GetAlignedPtr();
    map_key.data_type = data_type;
    map_key.format = output_tensor->GetFormat();
    map_key.shape = output_tensor->GetShape().GetDims();
    fuse_nodes[map_key].emplace_back(node);
    GELOGD("ConstantFuseSamePass, format %s, datatype %s, data_size %d, shape_size %zu. node name %s",
           TypeUtils::FormatToSerialString(map_key.format).c_str(),
           TypeUtils::DataTypeToSerialString(map_key.data_type).c_str(),
           map_key.data_size, map_key.shape.size(), node->GetName().c_str());
  }
  GELOGI("ConstantFuseSamePass, total_const_nums %d, insert_const_nums %d, fuse_nodes size is %zu.",
         total_const_nums, insert_const_nums, fuse_nodes.size());
}

Status ConstantFuseSamePass::MoveOutDataEdges(NodePtr &src_node, NodePtr &dst_node) {
  // key is node_name-in_index
  std::map<string, InDataAnchorPtr> src_out_node_to_indexs;
  GetOutDataNodeToIndexMap(src_node, src_out_node_to_indexs);
  if (src_out_node_to_indexs.empty()) {
    return SUCCESS;
  }

  std::map<string, InDataAnchorPtr> dst_out_node_to_indexs;
  GetOutDataNodeToIndexMap(dst_node, dst_out_node_to_indexs);

  auto dst_out_data_anchor = dst_node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(dst_out_data_anchor);
  auto src_out_data_anchor = src_node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(src_out_data_anchor);
  src_out_data_anchor->UnlinkAll();
  for (auto it = src_out_node_to_indexs.begin(); it != src_out_node_to_indexs.end(); ++it) {
    if (dst_out_node_to_indexs.count(it->first) > 0) {
      continue;  // exclusion of duplication
    }
    auto ret = dst_out_data_anchor->LinkTo(it->second);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Op:%s(%s) out index:0 link to op:%s(%s) in index:%d failed",
                        dst_node->GetName().c_str(), dst_node->GetType().c_str(),
                        it->second->GetOwnerNode()->GetName().c_str(), it->second->GetOwnerNode()->GetType().c_str(),
                        it->second->GetIdx());
      GELOGE(FAILED, "[Add][Edge] Op:%s(%s) out index:0 link to op:%s(%s) in index:%d failed",
             dst_node->GetName().c_str(), dst_node->GetType().c_str(),
             it->second->GetOwnerNode()->GetName().c_str(), it->second->GetOwnerNode()->GetType().c_str(),
             it->second->GetIdx());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ConstantFuseSamePass::FuseConstNodes(ComputeGraphPtr &graph,
                                            std::map<SameConstKey, std::vector<NodePtr>> &fuse_nodes) {
  for (auto iter = fuse_nodes.begin(); iter != fuse_nodes.end(); ++iter) {
    auto nodes = iter->second;
    size_t len = nodes.size();
    auto first_node = nodes.at(0);
    for (size_t i = 1; i < len; ++i) {
      auto node = nodes.at(i);

      GELOGI("Replace redundant const ndoe %s by %s", node->GetName().c_str(), first_node->GetName().c_str());
      // the const node which can be fused has none input(both data and control in)
      if (GraphUtils::MoveOutCtrlEdges(node, first_node) != SUCCESS) {
        return FAILED;
      }
      if (MoveOutDataEdges(node, first_node) != SUCCESS) {
        return FAILED;
      }
      if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                          node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
        GELOGE(FAILED, "[Remove][Node] %s(%s) Without Relink in graph:%s failed.",
               node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
