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

#include "graph/shape_refiner.h"

#include <memory>
#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <map>
#include <set>
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

#include "debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "external/graph/operator_factory.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
const char_t *const kPreOpInputShapeRange = "_pre_op_in_range";

const static std::set<std::string> kDummyContextOpTypes{ "Enter", "Switch", "RefSwitch", "StackPush", "StackPop" };
const static std::map<std::string, std::string> kGeLocalOpMapping{
    { "StreamMerge", "Merge" }, { "MemcpyAsync", "Identity" }
};
const int32_t kMaxRecursionDepth = 10;

bool IsOpWithSubgraph(const NodePtr &node) {
  const auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return false;
  }
  const auto subgraph_name = op_desc->GetSubgraphInstanceNames();
  return !subgraph_name.empty();
}

graphStatus UpdateOutputForMultiBatch(const ConstNodePtr &node,
                                      std::vector<std::vector<GeTensorDesc>> &ref_out_tensors) {
  // check sub_graph shape. Get max for update.
  for (size_t i = 0UL; i < ref_out_tensors.size(); ++i) {
    if (ref_out_tensors[i].empty()) {
      continue;
    }

    int64_t max_size = 0;
    size_t max_shape_index = 0UL;
    auto &ref_out_tensor = ref_out_tensors[i].at(0U);
    for (size_t j = 0UL; j < ref_out_tensors[i].size(); ++j) {
      auto &tensor = ref_out_tensors[i].at(j);
      if (ref_out_tensor.GetDataType() != tensor.GetDataType()) {
        REPORT_INNER_ERROR("E19999", "node[%s] does not support diff dtype among all ref output",
                           node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] does not support diff dtype among all ref output",
               node->GetName().c_str());
        return GRAPH_FAILED;
      }

      const auto shape = tensor.MutableShape();
      int64_t size = 1;
      for (const auto dim : shape.GetDims()) {
        if ((dim != 0) && (INT64_MAX / dim < size)) {
          REPORT_INNER_ERROR("E19999", "The shape:%s size overflow, node:%s",
                             shape.ToString().c_str(), node->GetName().c_str());
          GELOGE(PARAM_INVALID, "[Check][Overflow] The shape size overflow");
          return PARAM_INVALID;
        }
        size *= dim;
      }

      if (size > max_size) {
        max_size = size;
        max_shape_index = j;
      }
    }

    (void)node->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(i), ref_out_tensors[i].at(max_shape_index));
  }

  return GRAPH_SUCCESS;
}

graphStatus UpdateParentNodeForBranch(const ConstNodePtr &node,
                                      std::vector<std::vector<GeTensorDesc>> &ref_out_tensors) {
  GELOGD("Enter update parent node shape for class branch op process");
  if (node->GetOpDesc()->HasAttr(ATTR_NAME_BATCH_NUM)) {
    return UpdateOutputForMultiBatch(node, ref_out_tensors);
  }

  // check sub_graph shape.If not same ,do unknown shape process
  for (size_t i = 0UL; i < ref_out_tensors.size(); i++) {
    if (ref_out_tensors[i].empty()) {
      continue;
    }
    auto ref_out_tensor = ref_out_tensors[i].at(0U);
    ge::GeShape &ref_out_tensor_shape = ref_out_tensor.MutableShape();
    for (auto &tensor : ref_out_tensors[i]) {
      if (ref_out_tensor.GetDataType() != tensor.GetDataType()) {
        REPORT_INNER_ERROR("E19999", "node[%s] does not support diff dtype among all ref output, shape:%s",
                           node->GetName().c_str(), ref_out_tensor_shape.ToString().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] does not support diff dtype output", node->GetName().c_str());
        return GRAPH_FAILED;
      }
      const auto shape = tensor.MutableShape();
      if (shape.GetDims().size() != ref_out_tensor_shape.GetDims().size()) {
        GELOGD("node is %s, i : %zu, shape size: %lu, ref_out_tensor_shape size: %lu",
               node->GetName().c_str(), i, shape.GetShapeSize(), ref_out_tensor_shape.GetShapeSize());
        ref_out_tensor_shape = GeShape(UNKNOWN_RANK);
        break;
      }
      for (size_t j = 0UL; j < ref_out_tensor_shape.GetDims().size(); j++) {
        if (ref_out_tensor_shape.GetDim(j) == shape.GetDim(j)) {
          continue;
        }
        GELOGD("node is %s, i : %zu, j: %zu ,shape size: %lu, ref_out_tensor_shape size: %lu",
               node->GetName().c_str(), i, j, shape.GetShapeSize(), ref_out_tensor_shape.GetShapeSize());
        (void)ref_out_tensor_shape.SetDim(j, UNKNOWN_DIM);
      }
    }
    (void)node->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(i), ref_out_tensor);
  }
  return GRAPH_SUCCESS;
}

graphStatus UpdateParentNodeForWhile(const ConstNodePtr &node,
                                     std::vector<std::vector<GeTensorDesc>> &ref_data_tensors,
                                     std::vector<std::vector<GeTensorDesc>> &ref_out_tensors) {
  GELOGD("Enter update parent node shape for class while op process");
  if (ref_data_tensors.size() != ref_out_tensors.size()) {
    REPORT_INNER_ERROR("E19999", "op:%s(%s) input number[%zu] and output number[%zu] is not same!",
                       node->GetName().c_str(), node->GetType().c_str(),
                       ref_data_tensors.size(), ref_out_tensors.size());
    GELOGE(GRAPH_FAILED, "[Check][Param] while op [%s] input number[%zu] and output number[%zu] is not same!",
           node->GetName().c_str(), ref_data_tensors.size(), ref_out_tensors.size());
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < ref_data_tensors.size(); i++) {
    if (ref_out_tensors[i].size() != 1UL) {
      REPORT_INNER_ERROR("E19999", "while op, every output should only find one output tensor in all graph!");
      GELOGE(GRAPH_FAILED, "[Check][Param] while op, every output should only find one output tensor in all graph!");
      return GRAPH_FAILED;
    }
  }
  bool need_infer_again = false;
  // check input and output
  for (size_t i = 0UL; i < ref_out_tensors.size(); i++) {
    auto ref_out_tensor = ref_out_tensors[i].at(0U);
    const auto out_shape = ref_out_tensor.MutableShape();
    std::vector<std::pair<int64_t, int64_t>> data_shape_range;
    // ref_i's data and output tensor shape should be same
    for (auto &tensor : ref_data_tensors[i]) {
      if (ref_out_tensor.GetDataType() != tensor.GetDataType()) {
        REPORT_INNER_ERROR("E19999", "node[%s] does not support diff dtype or format among all ref output",
                           node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] does not support diff dtype or format output.",
               node->GetName().c_str());
        return GRAPH_FAILED;
      }
      auto data_shape = tensor.MutableShape();
      // input is dynamic, here use dim_num
      if (data_shape.GetDims() != out_shape.GetDims()) {
        GELOGI("After infer, While %s %zu output shape [%s] is not match with input shape [%s].Need infer again.",
               node->GetName().c_str(), i, out_shape.ToString().c_str(), data_shape.ToString().c_str());
        if (data_shape.GetDimNum() != out_shape.GetDimNum()) {
          ref_out_tensor.SetUnknownDimNumShape();
        } else {
          for (size_t j = 0; j < data_shape.GetDimNum(); ++j) {
            if (data_shape.GetDim(j) != out_shape.GetDim(j)) {
              if (data_shape.GetDim(j) != UNKNOWN_DIM) {
                // if input data is fix shape, output is different, need_infer_again
                need_infer_again = true;
              }
              (void)data_shape.SetDim(j, UNKNOWN_DIM);
            }
            // set shape rang of while, if dim is unknown ,set shape range as {1,-1}
            if (data_shape.GetDim(j) == UNKNOWN_DIM) {
              data_shape_range.emplace_back(std::make_pair(1, UNKNOWN_DIM));
            } else {
              data_shape_range.emplace_back(std::make_pair(data_shape.GetDim(j), data_shape.GetDim(j)));
            }
          }
          ref_out_tensor.SetShape(data_shape);
          (void)ref_out_tensor.SetShapeRange(data_shape_range);
        }
      }
    }
    (void)node->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(i), ref_out_tensor);
  }
  (void)AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_NEED_INFER_AGAIN, need_infer_again);
  return GRAPH_SUCCESS;
}

graphStatus UpdateSubGraphDataNodes(const ConstNodePtr &node) {
  // if infer again, update output of while into subgraph data node
  const auto op_desc = node->GetOpDesc();
  const auto sub_graph_names = op_desc->GetSubgraphInstanceNames();
  if (sub_graph_names.empty()) {
    return GRAPH_SUCCESS;
  }

  const auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  for (const auto &name : sub_graph_names) {
    const auto sub_graph = root_graph->GetSubgraph(name);
    if (sub_graph == nullptr) {
      REPORT_INNER_ERROR("E19999", "Can not find the subgrpah %s for node %s", name.c_str(), node->GetName().c_str());
      GE_LOGE("[Get][Graph] can not find the subgrpah %s for node %s", name.c_str(), node->GetName().c_str());
      return GRAPH_FAILED;
    }
    for (const auto &node_sub : sub_graph->GetDirectNode()) {
      if (node_sub->GetType() != DATA) {
        continue;
      }
      int32_t ref_i;
      const auto data_opdesc = node_sub->GetOpDesc();
      if (data_opdesc == nullptr) {
        REPORT_INNER_ERROR("E19999", "Invalid data node on the sub graph %s parent node %s, no OpDesc",
                           name.c_str(), node->GetName().c_str());
        GE_LOGE("[Get][OpDesc] Invalid data node on the sub graph %s parent node %s, no OpDesc",
                name.c_str(), node->GetName().c_str());
        return GRAPH_FAILED;
      }
      if (!AttrUtils::GetInt(data_opdesc, ATTR_NAME_PARENT_NODE_INDEX, ref_i)) {
        REPORT_INNER_ERROR("E19999", "Invalid data node on the sub graph %s parent node %s, no ref-index attribute",
                           name.c_str(), node->GetName().c_str());
        GE_LOGE("[Get][Int] Invalid data node on the sub graph %s parent node %s, no ref-index attribute",
                name.c_str(), node->GetName().c_str());
        return GRAPH_FAILED;
      }
      if (data_opdesc->HasAttr(ATTR_MBATCH_ORIGIN_INPUT_DIMS)) {
        continue;
      }
      auto input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(ref_i));
      if (input_desc == nullptr) {
        REPORT_INNER_ERROR("E19999", "The ref index(%d) on the data %s on the sub graph %s "
                           "parent node %s are incompatible, inputs num %u", ref_i, node_sub->GetName().c_str(),
                           name.c_str(), node->GetName().c_str(), node->GetAllInDataAnchorsSize());
        GE_LOGE("[Call][MutableInputDesc] The ref index(%d) on the data %s on the sub graph %s "
                "parent node %s are incompatible, inputs num %u", ref_i, node_sub->GetName().c_str(),
                name.c_str(), node->GetName().c_str(), node->GetAllInDataAnchorsSize());
        return GRAPH_FAILED;
      }
      GELOGI("Ref index is %d, input_desc dtype is %d, node name is %s", ref_i, input_desc->GetDataType(),
             node->GetName().c_str());

      // if need infer again, refresh subgraph input with output
      bool is_infer_again = false;
      (void)AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_NEED_INFER_AGAIN, is_infer_again);
      if (is_infer_again) {
        input_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(ref_i));
        if (input_desc == nullptr) {
          REPORT_INNER_ERROR("E19999", "The ref index(%d) on the data %s on the subgraph %s "
                             "parent node %s are incompatible, outputs num %u.", ref_i, node_sub->GetName().c_str(),
                             name.c_str(), node->GetName().c_str(), node->GetAllOutDataAnchorsSize());
          GELOGE(PARAM_INVALID, "[Call][MutableOutputDesc] The ref index(%d) on the data %s on the subgraph %s "
                 "parent node %s are incompatible, outputs num %u.", ref_i, node_sub->GetName().c_str(),
                 name.c_str(), node->GetName().c_str(), node->GetAllOutDataAnchorsSize());
        }
        GELOGD("Update input desc of data %s on the sub graph %s of node %s,output idx: %d from [%s] to [%s]",
               node_sub->GetName().c_str(),
               name.c_str(),
               node->GetName().c_str(),
               ref_i,
               data_opdesc->GetInputDescPtr(0U)->GetShape().ToString().c_str(),
               input_desc->GetShape().ToString().c_str());
      }

      auto ret = data_opdesc->UpdateInputDesc(0U, *input_desc);
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Failed to update input desc of data %s on the sub graph %s parent node %s",
                          node_sub->GetName().c_str(), name.c_str(), node->GetName().c_str());
        GE_LOGE("[Update][InputDesc] of data %s on the sub graph %s parent node %s failed",
                node_sub->GetName().c_str(), name.c_str(), node->GetName().c_str());
        return ret;
      }
      ret = data_opdesc->UpdateOutputDesc(0U, *input_desc);
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Failed to update output desc of data %s on the sub graph %s parent node %s",
                          node_sub->GetName().c_str(), name.c_str(), node->GetName().c_str());
        GE_LOGE("[Update][OutputDesc] of data %s on the sub graph %s parent node %s failed",
                node_sub->GetName().c_str(), name.c_str(), node->GetName().c_str());
        return ret;
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus FindSubgraphDataAndNetoutput(const std::shared_ptr<ComputeGraph> &sub_graph,
                                         NodePtr &netoutput, const ConstNodePtr &node,
                                         std::vector<std::vector<GeTensorDesc>> &ref_data_tensors) {
  auto sub_nodes = sub_graph->GetDirectNode();
  for (size_t i = sub_nodes.size(); i > 0UL; --i) {
    const auto sub_node = sub_nodes.at(i - 1UL);
    if (sub_node->GetType() == NETOUTPUT) {
      netoutput = sub_node;
    }
    if (sub_node->GetType() == DATA) {
      if (sub_node->GetOpDesc() == nullptr) {
        return GRAPH_FAILED;
      }

      int32_t ref_i;
      if (!AttrUtils::GetInt(sub_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, ref_i)) {
        REPORT_INNER_ERROR("E19999", "subgraph data node[%s] has no parent node!", sub_node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Get][Int] subgraph data node[%s] has no parent node!", sub_node->GetName().c_str());
        return GRAPH_FAILED;
      }
      if ((ref_i < 0) || (static_cast<uint32_t>(ref_i) >= node->GetAllInDataAnchorsSize())) {
        REPORT_INNER_ERROR("E19999", "data node[%s]'s ref index[%d] is not in range [0, %u)!",
                           sub_node->GetName().c_str(), ref_i, node->GetAllInDataAnchorsSize());
        GELOGE(GRAPH_FAILED, "[Check][Param] data node[%s]'s ref index[%d] is not in range [0, %u)!",
               sub_node->GetName().c_str(), ref_i, node->GetAllInDataAnchorsSize());
        return GRAPH_FAILED;
      }
      ref_data_tensors[static_cast<size_t>(ref_i)].emplace_back(sub_node->GetOpDesc()->GetOutputDesc(0U));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus UpdateParentNodeOutTensor(const ConstNodePtr &node) {
  const auto op_desc = node->GetOpDesc();
  const auto sub_graph_names = op_desc->GetSubgraphInstanceNames();
  if (sub_graph_names.empty()) {
    return GRAPH_SUCCESS;
  }

  std::vector<std::vector<GeTensorDesc>> ref_data_tensors(node->GetAllInDataAnchorsSize());
  std::vector<std::vector<GeTensorDesc>> ref_out_tensors(node->GetAllOutDataAnchorsSize());
  const auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());

  for (const auto &name : sub_graph_names) {
    auto sub_graph = root_graph->GetSubgraph(name);
    if (sub_graph == nullptr) {
      REPORT_INNER_ERROR("E19999", "Can not find the subgraph %s for node %s", name.c_str(), node->GetName().c_str());
      GE_LOGE("[Get][Subgraph] Can not find the subgraph %s for node %s", name.c_str(), node->GetName().c_str());
      return GRAPH_FAILED;
    }
    NodePtr netoutput = nullptr;
    const auto ret = FindSubgraphDataAndNetoutput(sub_graph, netoutput, node, ref_data_tensors);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
    if (netoutput == nullptr) {
      REPORT_INNER_ERROR("E19999", "No NetOutput node on sub graph %s, parent node %s",
                         name.c_str(), node->GetName().c_str());
      GE_LOGE("[Check][Param] No NetOutput node on sub graph %s, parent node %s",
              name.c_str(), node->GetName().c_str());
      return GRAPH_FAILED;
    }
    const auto netoutput_opdesc = netoutput->GetOpDesc();
    if (netoutput_opdesc == nullptr) {
      REPORT_INNER_ERROR("E19999", "Invalid NetOutput node on sub graph %s, parent node %s, no OpDesc on it",
                         name.c_str(), node->GetName().c_str());
      GE_LOGE("[Get][OpDesc] Invalid NetOutput node on sub graph %s, parent node %s, no OpDesc on it",
              name.c_str(), node->GetName().c_str());
      return GRAPH_FAILED;
    }
    for (auto &edge_anchor : netoutput->GetAllInDataAnchors()) {
      const auto edge_desc = netoutput_opdesc->MutableInputDesc(static_cast<uint32_t>(edge_anchor->GetIdx()));
      if (edge_desc == nullptr) {
        REPORT_INNER_ERROR("E19999", "Invalid NetOutput node on sub graph %s, parent node %s, "
                           "can not find input tensor %d",
                           name.c_str(), node->GetName().c_str(), edge_anchor->GetIdx());
        GE_LOGE("[Get][Tensor] Invalid NetOutput node on sub graph %s, parent node %s, can not find input tensor %d",
                name.c_str(), node->GetName().c_str(), edge_anchor->GetIdx());
        return GRAPH_FAILED;
      }
      GELOGI("Netoutput in anchor index is %d, input tensor dim is %zu",
             edge_anchor->GetIdx(), edge_desc->GetShape().GetDimNum());
      int32_t ref_i;
      if (!AttrUtils::GetInt(edge_desc, ATTR_NAME_PARENT_NODE_INDEX, ref_i)) {
        // if there is no ref index on the TensorDesc, it means the output data will be ignored outer.
        continue;
      }
      GELOGI("Parent node index of edge desc is %d", ref_i);
      if ((ref_i < 0) || (static_cast<uint32_t>(ref_i) >= node->GetAllOutDataAnchorsSize())) {
        return GRAPH_FAILED;
      }
      ref_out_tensors[static_cast<size_t>(ref_i)].emplace_back(*edge_desc);
    }
  }

  if (node->GetType() == WHILE) {
    return UpdateParentNodeForWhile(node, ref_data_tensors, ref_out_tensors);
  }
  return UpdateParentNodeForBranch(node, ref_out_tensors);
}

std::string Serial(const std::vector<int64_t> &dims) {
  std::string serial_string;
  serial_string += "[";
  for (const int64_t dim : dims) {
    serial_string += std::to_string(dim) + " ";
  }
  serial_string += "]";
  return serial_string;
}

void SerialShapeRange(const GeTensorDescPtr &desc, std::string &desc_str) {
  desc_str += "[";
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)desc->GetShapeRange(shape_range);
  for (const auto &pair : shape_range) {
    desc_str += "{";
    desc_str += std::to_string(pair.first) + "," + std::to_string(pair.second);
    desc_str += "},";
  }
  desc_str += "]";
  shape_range.clear();
  (void)desc->GetOriginShapeRange(shape_range);
  for (const auto &pair : shape_range) {
    desc_str += ",{";
    desc_str += std::to_string(pair.first) + "," + std::to_string(pair.second);
    desc_str += "},";
  }
}

graphStatus UpdateOpInputDesc(const ConstNodePtr &node_ptr) {
  GE_IF_BOOL_EXEC(node_ptr == nullptr, REPORT_INNER_ERROR("E19999", "param node_ptr is nullptr, check invalid.");
                  GELOGE(GRAPH_FAILED, "[Check][Param] node is null."); return GRAPH_FAILED);
  GE_IF_BOOL_EXEC(node_ptr->GetOpDesc() == nullptr,
                  REPORT_INNER_ERROR("E19999", "GetOpDesc failed, param node_ptr has no opdesc.");
                  GELOGE(GRAPH_FAILED, "[Get][OpDesc] op_desc is null."); return GRAPH_FAILED);
  for (const auto &in_anchor : node_ptr->GetAllInDataAnchors()) {
    const auto in_idx = in_anchor->GetIdx();
    const auto peer_out_data_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_data_anchor == nullptr) {
      continue;
    }
    const auto peer_out_data_node = peer_out_data_anchor->GetOwnerNode();
    if (peer_out_data_node == nullptr || peer_out_data_node->GetOpDesc() == nullptr) {
      continue;
    }
    const int32_t peer_out_idx = peer_out_data_anchor->GetIdx();
    const auto peer_out_desc = peer_out_data_node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(peer_out_idx));

    // check shape and dtype continuity. do not stop process
    const auto in_desc = node_ptr->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(in_idx));
    if (in_desc == nullptr) {
      continue;
    }
    const auto in_shape = in_desc->MutableShape().GetDims();
    const auto in_dtype = in_desc->GetDataType();
    const auto peer_out_shape = peer_out_desc->MutableShape().GetDims();
    const auto peer_out_dtype = peer_out_desc->GetDataType();
    if (peer_out_dtype != in_dtype) {
      GELOGW("[Update][InputDesc] current node [%s] [%d]\'th in_dtype is [%s].peer output node [%s] [%d]\'th "
             "output_dtype is [%s]. The two dtype should be same! Please check graph and fix it",
             node_ptr->GetName().c_str(), in_idx, TypeUtils::DataTypeToSerialString(in_dtype).c_str(),
             peer_out_data_node->GetName().c_str(), peer_out_idx,
             TypeUtils::DataTypeToSerialString(peer_out_dtype).c_str());
    } else if ((!in_shape.empty()) && (in_shape != peer_out_shape)) {
      const std::string in_shape_str = Serial(in_shape);
      const std::string peer_out_shape_str = Serial(peer_out_shape);
      GELOGW("[Update][InputDesc] current node [%s] [%d]\'th in_shape is [%s].peer output node [%s] [%d]\'th "
             "output_shape is [%s]. The two shape should be same! Please check graph and fix it",
             node_ptr->GetName().c_str(), in_idx, in_shape_str.c_str(),
             peer_out_data_node->GetName().c_str(), peer_out_idx, peer_out_shape_str.c_str());
    }
    // refresh current node input desc
    in_desc->SetOriginShape(peer_out_desc->GetOriginShape());
    in_desc->SetShape(peer_out_desc->MutableShape());
    in_desc->SetDataType(peer_out_desc->GetDataType());
    in_desc->SetOriginDataType(peer_out_desc->GetOriginDataType());
    if (peer_out_desc->MutableShape().GetDims() != UNKNOWN_RANK) {
      std::vector<std::pair<int64_t, int64_t>> shape_range;
      (void)peer_out_desc->GetShapeRange(shape_range);
      (void)in_desc->SetShapeRange(shape_range);
    }
    std::vector<int64_t> pre_op_in_range;
    if (ge::AttrUtils::GetListInt(*peer_out_desc, kPreOpInputShapeRange, pre_op_in_range)) {
      (void)ge::AttrUtils::SetListInt(*in_desc, kPreOpInputShapeRange, pre_op_in_range);
    }
    ge::TensorUtils::SetRealDimCnt(*in_desc,
                                   static_cast<uint32_t>(peer_out_desc->MutableShape().GetDims().size()));
  }
  return GRAPH_SUCCESS;
}
}  // namespace
void ShapeRefiner::PrintInOutTensorShape(const ge::NodePtr &node, const std::string &phase) {
  if (!IsLogEnable(GE, DLOG_DEBUG)) {
    return;
  }
  const ge::OpDescPtr op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(op_desc == nullptr, REPORT_INNER_ERROR("E19999", "node has no opdesc, check invalid");
                  GELOGE(GRAPH_FAILED, "[Get][OpDesc] op_desc is null."); return);
  std::stringstream ss;
  ss << "{";
  int32_t in_idx = 0;
  int32_t out_idx = 0;
  for (const auto &input_desc : op_desc->GetAllInputsDescPtr()) {
    if (input_desc == nullptr) {
      in_idx++;
      continue;
    }
    if (in_idx > 0) {
      ss << "    ";
    }
    ss << "input_" << in_idx << " " << "tensor: [";
    ss << "(shape:[" << input_desc->MutableShape().ToString() << "]),";
    ss << "(format:" << TypeUtils::FormatToSerialString(input_desc->GetFormat()) << "),";
    ss << "(dtype:" << TypeUtils::DataTypeToSerialString(input_desc->GetDataType()) << "),";
    ss << "(origin_shape:" << input_desc->GetOriginShape().ToString() << "),";
    ss << "(origin_format:" << TypeUtils::FormatToSerialString(input_desc->GetOriginFormat()) << "),";
    ss << "(origin_dtype:" << TypeUtils::DataTypeToSerialString(input_desc->GetOriginDataType()) << "),";
    std::string range_str;
    SerialShapeRange(input_desc, range_str);
    ss << "(shape_range:" << range_str << ")]";
    in_idx++;
  }
  for (const auto &output_desc : op_desc->GetAllOutputsDescPtr()) {
    if (output_desc == nullptr) {
      out_idx++;
      continue;
    }
    ss << "    ";
    ss << "output_" << out_idx << " " << "tensor: [";
    ss << "(shape:[" << output_desc->MutableShape().ToString() << "]),";
    ss << "(format:" << TypeUtils::FormatToSerialString(output_desc->GetFormat()) << "),";
    ss << "(dtype:" << TypeUtils::DataTypeToSerialString(output_desc->GetDataType()) << "),";
    ss << "(origin_shape:" << output_desc->GetOriginShape().ToString() << "),";
    ss << "(origin_format:" << TypeUtils::FormatToSerialString(output_desc->GetOriginFormat()) << "),";
    ss << "(origin_dtype:" << TypeUtils::DataTypeToSerialString(output_desc->GetOriginDataType()) << "),";
    std::string range_str;
    SerialShapeRange(output_desc, range_str);
    ss << "(shape_range:" << range_str << ")]";
    out_idx++;
  }
  ss << "}";
  GELOGD("Shape dump [%s], Node name[%s], type[%s]. %s", phase.c_str(), node->GetName().c_str(),
         node->GetType().c_str(), ss.str().c_str());
}

namespace {
thread_local std::unordered_map<NodePtr, InferenceContextPtr> context_map;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
void ShapeRefiner::ClearContextMap() {
  context_map.clear();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
void ShapeRefiner::PushToContextMap(const NodePtr &node, const InferenceContextPtr &inference_context) {
  (void)context_map.emplace(node, inference_context);
}

Status GetOutNodesByParentNodeOutIndex(const NodePtr &parent_node, const int32_t out_idx,
                                       std::map<NodePtr, int32_t> &out_nodes, const int32_t depth) {
  if (depth > kMaxRecursionDepth) {
    REPORT_CALL_ERROR("E19999", "Exceed max recursion depth: %d.", kMaxRecursionDepth);
    GELOGE(FAILED, "[Validate][Depth] Exceed max recursion depth: %d.", kMaxRecursionDepth);
    return FAILED;
  }
  out_nodes.clear();
  if (!IsOpWithSubgraph(parent_node)) {
    return SUCCESS;
  }
  GELOGD("Node: %s, out index: %d.", parent_node->GetName().c_str(), out_idx);
  auto subgraph_output_nodes = NodeUtils::GetSubgraphOutputNodes(*parent_node);
  for (const auto &netoutput : subgraph_output_nodes) {
    GE_CHECK_NOTNULL(netoutput);
    const auto output_desc = netoutput->GetOpDesc();
    GE_CHECK_NOTNULL(output_desc);
    for (const auto &in_data_anchor : netoutput->GetAllInDataAnchors()) {
      GE_CHECK_NOTNULL(in_data_anchor);
      const auto in_desc = output_desc->MutableInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
      GE_CHECK_NOTNULL(in_desc);
      int32_t ref = 0;
      if (AttrUtils::GetInt(in_desc, ATTR_NAME_PARENT_NODE_INDEX, ref) && (ref == out_idx)) {
        const auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
        GE_CHECK_NOTNULL(peer_out_data_anchor);
        auto peer_out_data_node = peer_out_data_anchor->GetOwnerNode();
        if (IsOpWithSubgraph(peer_out_data_node)) {
          std::map<NodePtr, int32_t> tmp_nodes;
          if (GetOutNodesByParentNodeOutIndex(peer_out_data_node, peer_out_data_anchor->GetIdx(), tmp_nodes,
                                              depth + 1) != SUCCESS) {
            REPORT_CALL_ERROR("E19999", "Get out nodes of %s by index failed.", peer_out_data_node->GetName().c_str());
            GELOGE(FAILED, "[Get][Outnodes] of %s by index failed.", peer_out_data_node->GetName().c_str());
            return FAILED;
          }
          out_nodes.insert(tmp_nodes.begin(), tmp_nodes.end());
        } else {
          (void)out_nodes.emplace(peer_out_data_node, peer_out_data_anchor->GetIdx());
        }
        GELOGI("Peer node: %s, out index: %d, ref: %d.", peer_out_data_node->GetName().c_str(),
               peer_out_data_anchor->GetIdx(), ref);
      }
    }
  }
  return SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus ShapeRefiner::GetRealInNodesAndIndex(NodePtr &input_node, int32_t &output_idx,
                                                 std::map<NodePtr, int32_t> &nodes_idx) {
  auto op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  while (input_node->GetType() == DATA && op_desc->HasAttr(ATTR_NAME_PARENT_NODE_INDEX)) {
    int32_t ref_i = 0;
    (void)AttrUtils::GetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, ref_i);
    const auto owner_graph = input_node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(owner_graph);
    const auto parent_node = owner_graph->GetParentNode();
    GE_CHECK_NOTNULL(parent_node);
    const auto in_data_anchor = parent_node->GetInDataAnchor(ref_i);
    GE_CHECK_NOTNULL(in_data_anchor);
    const auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_data_anchor);
    output_idx = peer_out_data_anchor->GetIdx();
    input_node = peer_out_data_anchor->GetOwnerNode();
    op_desc = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GELOGD("In node[%s], type[%s], ref[%d].", input_node->GetName().c_str(), input_node->GetType().c_str(), ref_i);
  }

  if (IsOpWithSubgraph(input_node)) {
    if (GetOutNodesByParentNodeOutIndex(input_node, output_idx, nodes_idx, 0) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Get outnodes of %s by parent node out index failed.", input_node->GetName().c_str());
      GELOGE(FAILED, "[Get][Outnodes] of %s by parent node out index failed.", input_node->GetName().c_str());
      return FAILED;
    }
    GELOGI("Out node num: %zu.", nodes_idx.size());
  }
  if (nodes_idx.empty()) {
    (void)nodes_idx.emplace(input_node, output_idx);
  }
  return SUCCESS;
}


GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus ShapeRefiner::CreateInferenceContext(const NodePtr &node, InferenceContextPtr &inference_context) {
  return CreateInferenceContext(node, nullptr, inference_context);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus ShapeRefiner::CreateInferenceContext(const NodePtr &node, ResourceContextMgr *resource_context_mgr,
                                                 InferenceContextPtr &inference_context) {
  GE_CHECK_NOTNULL(node);
  inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create(resource_context_mgr));
  GE_CHECK_NOTNULL(inference_context);
  const auto all_in_data_anchors = node->GetAllInDataAnchors();
  std::vector<std::vector<ShapeAndType>> input_shapes_and_types(all_in_data_anchors.size());
  std::vector<std::string> marks;

  bool has_input_shapes_and_types = false;
  for (const auto &in_anchor : all_in_data_anchors) {
    GE_CHECK_NOTNULL(in_anchor);
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }

    auto input_node = out_anchor->GetOwnerNode();
    auto output_idx = out_anchor->GetIdx();
    std::map<NodePtr, int32_t> input_nodes_2_out_idx;
    if (GetRealInNodesAndIndex(input_node, output_idx, input_nodes_2_out_idx) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Failed to get real in nodes and index, node:%s", node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Get][InNodesAndIndex] of node[%s] failed.", node->GetName().c_str());
      return GRAPH_FAILED;
    }

    const auto input_idx = in_anchor->GetIdx();
    for (const auto &node_idx : input_nodes_2_out_idx) {
      const auto in_node = node_idx.first;
      GELOGD("Input node[%s], type[%s], context_map size[%zu].", in_node->GetName().c_str(), in_node->GetType().c_str(),
             context_map.size());
      const auto iter = context_map.find(in_node);
      if (iter != context_map.end()) {
        const auto &src_context = iter->second;
        GE_CHECK_NOTNULL(src_context);
        GELOGD("node:%s get %ld marks from node:%s",
               node->GetName().c_str(), src_context->GetMarks().size(), in_node->GetName().c_str());
        for (auto mark : src_context->GetMarks()) {
          if (marks.empty()) {
            marks.emplace_back(mark);
          }
        }
        const auto output_idx = node_idx.second;
        const auto output_shape_and_type = src_context->GetOutputHandleShapesAndTypes();
        if (output_idx < static_cast<int32_t>(output_shape_and_type.size())) {
          GELOGI("Add shape and type from %s:%d to %s:%d", in_node->GetName().c_str(), output_idx,
                 node->GetName().c_str(), input_idx);
          input_shapes_and_types[static_cast<size_t>(input_idx)] =
              output_shape_and_type[static_cast<size_t>(output_idx)];
          has_input_shapes_and_types = true;
        } else {
          GELOGI("[%s] Output out of range. index = %d, size = %zu", node->GetName().c_str(), output_idx,
                 output_shape_and_type.size());
        }
      }
    }
  }

  if (has_input_shapes_and_types) {
    inference_context->SetInputHandleShapesAndTypes(std::move(input_shapes_and_types));
  }
  GELOGD("Node: %s, marks size: %zu.", node->GetName().c_str(), marks.size());
  inference_context->SetMarks(marks);

  return SUCCESS;
}

graphStatus ShapeRefiner::InferShapeAndType(const ConstNodePtr &node, Operator &op) {
  return InferShapeAndType(node, op, true);
}

graphStatus ShapeRefiner::InferShapeAndType(const ConstNodePtr &node, Operator &op, const bool before_subgraph) {
  const auto op_desc = node->GetOpDesc();
  const auto &op_type = op_desc->GetType();

  graphStatus ret;
  if (before_subgraph) {
    ret = UpdateSubGraphDataNodes(node);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
  }
  // Get infer func and execute
  ret = op_desc->CallInferFunc(op);
  if (ret == GRAPH_PARAM_INVALID) {
    // Op ir no infer func, try to get infer func from operator factory
    const auto node_op = ge::OperatorFactory::CreateOperator("node_op", op_desc->GetType());
    if (node_op.IsEmpty()) {
      GELOGW("[InferShape][Check] Get op from OperatorFactory failed, type: %s", op_type.c_str());
      return ret;
    }

    GELOGD("get op from OperatorFactory success. opType: %s", op_type.c_str());
    const auto temp_op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
    node_op.BreakConnect();
    if (temp_op_desc == nullptr) {
      REPORT_CALL_ERROR("E19999", "GetOpDescFromOperator failed, return nullptr.");
      GELOGE(GRAPH_FAILED, "[Get][OpDesc] temp op desc is null");
      return GRAPH_FAILED;
    }
    if (!op_desc->UpdateInputName(temp_op_desc->GetAllInputName())) {
      GELOGW("[InferShape][UpdateInputName] Update input name failed");
      for (const auto &out_desc : op_desc->GetAllOutputsDescPtr()) {
        if (out_desc != nullptr && out_desc->GetShape().GetDims().empty()) {
          break;
        }
        return GRAPH_SUCCESS;
      }
    }
    if (!op_desc->UpdateOutputName(temp_op_desc->GetAllOutputName())) {
      GELOGW("[InferShape][UpdateOutputName] Update output name failed");
    }
    op_desc->AddInferFunc(temp_op_desc->GetInferFunc());
    ret = op_desc->CallInferFunc(op);
    GELOGI("op CallInferFunc second. ret: %u", ret);
  }
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  if (!before_subgraph) {
    return UpdateParentNodeOutTensor(node);
  }
  return GRAPH_SUCCESS;
}

graphStatus ShapeRefiner::DoInferShapeAndTypeForRunning(const ConstNodePtr &node, Operator &op,
                                                        const bool before_subgraph) {
  const auto op_desc = node->GetOpDesc();
  const auto origin_type = NodeUtils::GetNodeType(*node);

  graphStatus ret;
  if (before_subgraph) {
    ret = UpdateSubGraphDataNodes(node);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
  }

  // Create InferenceContext to avoid null pointer access.
  if (kDummyContextOpTypes.count(origin_type) > 0U) {
    GELOGD("Set InferenceContext for node [%s]", op_desc->GetName().c_str());
    op.SetInferenceContext(std::shared_ptr<InferenceContext>(InferenceContext::Create()));
  }

  // Get infer func and execute
  ret = op_desc->CallInferFunc(op);
  if (ret == GRAPH_PARAM_INVALID) {
    GELOGD("NodeUtils::GetNodeType return value is: [%s]", origin_type.c_str());
    const auto it = kGeLocalOpMapping.find(origin_type);
    const auto infer_func =
        OperatorFactoryImpl::GetInferShapeFunc(it == kGeLocalOpMapping.end() ? origin_type : it->second);
    if (infer_func == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to Get InferFunc. type is %s", origin_type.c_str());
      GELOGE(GRAPH_FAILED, "[Get][InferFunc] failed. type is %s", origin_type.c_str());
      return GRAPH_FAILED;
    }
    op_desc->AddInferFunc(infer_func);
    ret = op_desc->CallInferFunc(op);
    GELOGI("op CallInferFunc second. ret: %u", ret);
  }
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  if (!before_subgraph) {
    return UpdateParentNodeOutTensor(node);
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus ShapeRefiner::InferShapeAndType(const NodePtr &node) {
  return InferShapeAndType(node, true);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus ShapeRefiner::InferShapeAndTypeForRunning(const NodePtr &node, Operator &op, const bool before_subgraph) {
  GE_CHECK_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  std::vector<ge::DataType> temp_dtype;
  for (auto &tensor_desc: op_desc->GetAllOutputsDescPtr()) {
      temp_dtype.emplace_back(tensor_desc->GetDataType());
  }
  PrintInOutTensorShape(node, "before_infershape when running");

  const graphStatus status = DoInferShapeAndTypeForRunning(node, op, before_subgraph);
  if ((status == GRAPH_PARAM_INVALID) || (status == GRAPH_SUCCESS)) {
    // ensure the dtype is not changed after infershape in running
    const auto after_opdesc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(after_opdesc == nullptr, REPORT_INNER_ERROR("E19999", "param node has no opdesc, check invalid.");
                    GELOGE(GRAPH_FAILED, "[Get][OpDesc]  after_opdesc is null."); return GRAPH_FAILED);
    auto all_output_tensor = after_opdesc->GetAllOutputsDescPtr();
    for (size_t i = 0UL; i < all_output_tensor.size(); ++i) {
      if (all_output_tensor.at(i)->GetDataType() != temp_dtype[i]) {
        GELOGD("Op %s output %zu need reset dtype,original dtype is %s, new dtype is %s",
               node->GetName().c_str(), i,
               TypeUtils::DataTypeToSerialString(all_output_tensor.at(i)->GetDataType()).c_str(),
               TypeUtils::DataTypeToSerialString(temp_dtype[i]).c_str());
        all_output_tensor.at(i)->SetDataType(temp_dtype[i]);
      }
    }
    PrintInOutTensorShape(node, "after_infershape when running");
    return GRAPH_SUCCESS;
  } else {
    REPORT_CALL_ERROR("EZ9999", "%s(%s) call infer function failed.",
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(GRAPH_FAILED, "[Call][InferFunction] failed, node:%s(%s).",
           node->GetName().c_str(), node->GetType().c_str());
    return GRAPH_FAILED;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus ShapeRefiner::UpdateInputOutputDesc(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    auto output_tensor = op_desc->MutableOutputDesc(static_cast<uint32_t>(out_anchor->GetIdx()));
    GE_IF_BOOL_EXEC(output_tensor == nullptr, continue);
    GE_IF_BOOL_EXEC(output_tensor->MutableShape().GetDims().empty(),
                    output_tensor->SetOriginShape(output_tensor->GetShape()));

    ge::TensorUtils::SetRealDimCnt(*output_tensor, static_cast<uint32_t>(output_tensor->GetOriginShape().GetDims()
    .size()));
    output_tensor->SetOriginDataType(output_tensor->GetDataType());
    // set output origin shape range
    std::vector<std::pair<int64_t, int64_t>> range;
    (void)output_tensor->GetShapeRange(range);
    (void)output_tensor->SetOriginShapeRange(range);
    GELOGD("node name is %s, origin shape is %ld, origin format is %s, origin data type is %s",
           node->GetName().c_str(), output_tensor->GetOriginShape().GetShapeSize(),
           TypeUtils::FormatToSerialString(output_tensor->GetOriginFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor->GetOriginDataType()).c_str());
  }
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    const auto input_tensor = op_desc->MutableInputDesc(static_cast<uint32_t>(in_anchor->GetIdx()));
    GE_IF_BOOL_EXEC(input_tensor == nullptr, continue);

    // set input origin shape range
    std::vector<std::pair<int64_t, int64_t>> range;
    (void)input_tensor->GetShapeRange(range);
    (void)input_tensor->SetOriginShapeRange(range);
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus ShapeRefiner::PostProcessAfterInfershape(const NodePtr &node, const Operator &op,
                                                     const bool is_unknown_graph) {
  GE_CHECK_NOTNULL(node);
  if (is_unknown_graph) {
    PrintInOutTensorShape(node, "after_infershape when running");
    return GRAPH_SUCCESS;
  }

  if (UpdateInputOutputDesc(node) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Update input and output desc of %s failed.", node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Update][TensorDesc] Update input and output desc of %s failed.", node->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (!is_unknown_graph) {
    auto ctx_after_infer = op.GetInferenceContext();
    if (ctx_after_infer != nullptr) {
      GELOGD("[%s] after infershape. mark:%zu", node->GetName().c_str(), ctx_after_infer->GetMarks().size());
      if ((!ctx_after_infer->GetOutputHandleShapesAndTypes().empty()) || (!ctx_after_infer->GetMarks().empty())) {
        GELOGD("[%s] set inference context after. mark:%zu", node->GetName().c_str(),
               ctx_after_infer->GetMarks().size());
        (void)context_map.emplace(node, ctx_after_infer);
      }
    }
  }
  PrintInOutTensorShape(node, "after_infershape");

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus ShapeRefiner::InferShapeAndType(const NodePtr &node, const bool before_subgraph) {
  GE_CHECK_NOTNULL(node);
  const bool is_unknown_graph = node->GetOwnerComputeGraph()->GetGraphUnknownFlag();
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  // some op can not infershape twice such as aipp
  const bool need_update_input = (!is_unknown_graph) && (!op_desc->HasAttr("has_infered_verified"));
  if (need_update_input) {
    const auto status = UpdateOpInputDesc(node);
    if (status != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "update op input_desc failed! ret:%d, node:%s", status, node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Update][OpInputDesc] failed! ret:%d", status);
      return status;
    }
  }

  if (node->Verify() != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("EZ9999", "Verifying %s(%s) failed.", node->GetName().c_str(), node->GetType().c_str());
    GELOGE(GRAPH_FAILED, "[Call][Verify] Verifying %s(%s) failed.", node->GetName().c_str(), node->GetType().c_str());
    return GRAPH_FAILED;
  }
  PrintInOutTensorShape(node, "before_infershape");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);  // do not need runtime context

  if (!is_unknown_graph) {
    InferenceContextPtr inference_context;
    if (CreateInferenceContext(node, inference_context) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "CreateInferenceContext of %s failed.", node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Create][Context] CreateInferenceContext of %s failed.", node->GetName().c_str());
      return GRAPH_FAILED;
    }
    GE_CHECK_NOTNULL(inference_context);
    GELOGD("create context for node:%s, marks %zu", node->GetName().c_str(), inference_context->GetMarks().size());
    op.SetInferenceContext(inference_context);
  }

  const graphStatus status = InferShapeAndType(node, op, before_subgraph);
  const bool check_status_valid = (status == GRAPH_PARAM_INVALID) || (status == GRAPH_SUCCESS);
  if (!check_status_valid) {
    REPORT_CALL_ERROR("EZ9999", "%s(%s) call infer function failed.",
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(GRAPH_FAILED, "[Call][InferFunction] failed, node:%s(%s).",
           node->GetName().c_str(), node->GetType().c_str());
    return GRAPH_FAILED;
  }

  return PostProcessAfterInfershape(node, op, is_unknown_graph);
}
}  // namespace ge
