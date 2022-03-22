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

#include "graph/passes/buffer_pool_memory_pass.h"

#include <string>
#include <vector>
#include "common/omg_util.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/math/math_util.h"

namespace ge {
namespace {
const size_t kBufferPoolNodeInSize = 1;
const size_t kBufferPoolNodeOutSize = 1;
} // namespace

Status BufferPoolMemoryPass::Run(ComputeGraphPtr graph) {
  if (graph == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Graph]Graph is nullptr");
    REPORT_INNER_ERROR("E19999", "Input graph is nullptr");
    return PARAM_INVALID;
  }
  // The cache prefetching scheme is developed for very large models, which gets the weight data in advance
  // and allocates it to a special memory pool. When the large model is dynamic shape, it need to go through
  // the executor flow and is not allocated memory statically. This is another development point, so we will
  // skip the dynamic shape model processing here.
  if (graph->GetParentGraph() != nullptr || graph->GetGraphUnknownFlag()) {
    return SUCCESS;
  }
  if (!IsBufferPoolMemEnable(graph)) {
    GELOGD("[Check][Enable]Buffer pool memory is not enable, graph:%s.", graph->GetName().c_str());
    return SUCCESS;
  }
  Status ret = graph->TopologicalSorting();
  if (ret != SUCCESS) {
    GELOGE(ret, "[TopologicalSort][Graph]Graph name:%s.", graph->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Failed to topological sort for graph:%s.", graph->GetName().c_str());
    return ret;
  }

  ret = CopyOutForMultiUsedOutput(graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Copy][Output]Graph:%s.", graph->GetName().c_str());
    return FAILED;
  }

  ret = GetBufferPoolAndPeerCalcNodes(graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][BufferPoolNode]Graph:%s.", graph->GetName().c_str());
    return FAILED;
  }
  if (calc_nodes_.empty()) {
    GELOGE(FAILED, "[Check][BufferPoolNode]Graph:%s.", graph->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "All Buffer pool nodes are isolated nodes in graph:%s.", graph->GetName().c_str());
    return FAILED;
  }
  ret = AllocateAllBufferPoolSpace();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Alloc][BufferPoolMem]Graph:%s.", graph->GetName().c_str());
    return FAILED;
  }

  ret = SetResultOfMemoryAndEvent();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Set][Result]Graph:%s.", graph->GetName().c_str());
    return FAILED;
  }
  ret = graph->TopologicalSorting();
  if (ret != SUCCESS) {
    GELOGE(ret, "[TopologicalSort][Graph]Graph name:%s.", graph->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Failed to topological sort for graph:%s.", graph->GetName().c_str());
    return ret;
  }
  return SUCCESS;
}

void BufferPoolMemoryPass::ClearQueue(std::queue<std::pair<std::string, uint32_t>> &q) {
  while (!q.empty()) {
    q.pop();
  }
}

Status BufferPoolMemoryPass::IsBufferPoolMemEnable(const ComputeGraphPtr &graph) {
  for (NodePtr &node : graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    if (op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_ID) && op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_SIZE)) {
      return true;
    }
  }
  return false;
}

Status BufferPoolMemoryPass::CheckBufferPoolSize(int64_t total_size, int64_t pool_id, int64_t buffer_pool_size,
                                                 std::unordered_map<int64_t, int64_t> &calc_total_size) {
  auto iter = calc_total_size.find(pool_id);
  if (iter == calc_total_size.end()) {
    calc_total_size[pool_id] = total_size;
  } else {
    FMK_INT64_ADDCHECK(calc_total_size[pool_id], total_size);
    calc_total_size[pool_id] += total_size;
  }
  if (calc_total_size[pool_id] > buffer_pool_size) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]The memory required at the same is greater than buffer pool size, "
          "pool id:%ld, pool size:%ld, required size:%ld.", pool_id, buffer_pool_size, calc_total_size[pool_id]);
    REPORT_INNER_ERROR("E19999", "The memory required at the same is greater than buffer pool size, pool id:%ld,"
                       " pool size:%ld, required size:%ld.", pool_id, buffer_pool_size, calc_total_size[pool_id]);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::TryToFixNodeOrder(NodePtr &pre_node, NodePtr &curr_node, bool &not_change) {
  auto pre_node_graph = pre_node->GetOwnerComputeGraph();
  auto curr_node_graph = curr_node->GetOwnerComputeGraph();
  std::string pre_node_stream_label;
  (void) AttrUtils::GetStr(pre_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, pre_node_stream_label);
  std::string curr_node_stream_label;
  (void) AttrUtils::GetStr(curr_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, curr_node_stream_label);
  not_change = true;
  if ((pre_node_graph == curr_node_graph) && (pre_node_stream_label == pre_node_stream_label)) {
    // Same subgraph, including simultaneously in the root graph.
    auto ret = ge::GraphUtils::AddEdge(pre_node->GetOutControlAnchor(), curr_node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Add][Edge]Src:%s, dst:%s.", pre_node->GetName().c_str(), curr_node->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "Failed to add ctrl edge from %s to %s.",
                        pre_node->GetName().c_str(), curr_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    not_change = false;
  } else if (pre_node_graph->GetParentGraph() == curr_node_graph->GetParentGraph() &&
             pre_node_graph->GetParentNode() != nullptr && curr_node_graph->GetParentNode() != nullptr) {
    // Two nodes are located on different child graphs of different parent nodes.
    auto pre_node_parent_op_desc = pre_node_graph->GetParentNode()->GetOpDesc();
    auto curr_node_parent_op_desc = curr_node_graph->GetParentNode()->GetOpDesc();
    GE_CHECK_NOTNULL(pre_node_parent_op_desc);
    GE_CHECK_NOTNULL(curr_node_parent_op_desc);
    // The parent node dependency is correct to ensure that the child node dependency,
    // there is no need to add control edges.
    if (pre_node_parent_op_desc->GetId() > curr_node_parent_op_desc->GetId()) {
      GELOGE(INTERNAL_ERROR, "[Check][Dependency]Invalid dependency, pre node:%s, curr node:%s.",
             pre_node->GetName().c_str(), curr_node->GetName().c_str());
      REPORT_INNER_ERROR("E19999", "Invalid dependency, pre node:%s, curr node:%s.",
                         pre_node->GetName().c_str(), curr_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    GELOGI("[Check][Dependency]The two nodes are located in sub graphs of different parent nodes and meet the "
           "dependency relationship. pre:%s, curr:%s.", pre_node->GetName().c_str(), curr_node->GetName().c_str());
  } else {
    GELOGE(INTERNAL_ERROR, "[Check][Dependency]Invalid dependency, pre node:%s, curr node:%s.",
           pre_node->GetName().c_str(), curr_node->GetName().c_str());
    REPORT_INNER_ERROR("E19999", "Invalid dependency, pre node:%s, curr node:%s.",
                       pre_node->GetName().c_str(), curr_node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::InsertMemCpyNodeAfter(ComputeGraphPtr &graph, NodePtr &node) {
  auto out_anchor = node->GetOutDataAnchor(kBufferPoolNodeOutIndex);
  OpDescBuilder op_desc_builder(node->GetName() + "_memcpy_async", MEMCPYASYNC);
  auto mem_copy_op = op_desc_builder.AddInput("x", node->GetOpDesc()->GetOutputDesc(kBufferPoolNodeOutIndex))
    .AddOutput("y", node->GetOpDesc()->GetOutputDesc(kBufferPoolNodeOutIndex))
    .Build();
  std::string batch_label;
  bool get_attr = AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, batch_label);
  if (get_attr && !batch_label.empty()) {
    (void) AttrUtils::SetStr(mem_copy_op, ATTR_NAME_STREAM_LABEL, batch_label);
  }
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  std::vector<InDataAnchorPtr> in_anchors(peer_in_anchors.begin(), peer_in_anchors.end());
  if (GraphUtils::InsertNodeAfter(out_anchor, in_anchors, graph->AddNode(mem_copy_op)) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "[Insert][Node] Node:%s.", node->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Failed to insert mem copy node after %s.", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::CopyOutForMultiUsedOutput(ComputeGraphPtr &graph) {
  bool changed = false;
  for (NodePtr &node : graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    bool use_buffer_pool = op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_ID) && op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_SIZE);
    if (use_buffer_pool) {
      if ((node->GetInDataNodes().size() == kBufferPoolNodeInSize) &&
          (node->GetOutDataNodes().size() == kBufferPoolNodeOutSize)) {
        continue;
      } else if ((node->GetAllInDataAnchors().size() == kBufferPoolNodeInSize) &&
                 (node->GetAllOutDataAnchors().size() == kBufferPoolNodeOutSize)) {
        // A prefetching output is used in multiple places. Copy one so that the prefetching node remains
        // single input and single output.
        if (InsertMemCpyNodeAfter(graph, node) != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "[Insert][MemCpy]Node:%s.", node->GetName().c_str());
          REPORT_INNER_ERROR("E19999", "Failed to insert mem copy node after %s.", node->GetName().c_str());
          return INTERNAL_ERROR;
        }
        changed = true;
        GELOGI("[Insert][Node]Insert mem copy node after %s.", node->GetName().c_str());
      } else {
        GELOGE(PARAM_INVALID, "[Check][InputOutput]Only support single input and single output, "
               "node:%s.", node->GetName().c_str());
        REPORT_INNER_ERROR("E19999", "Only support single input and single output, node:%s.", node->GetName().c_str());
        return PARAM_INVALID;
      }
    }
  }
  if (changed) {
    Status ret = graph->TopologicalSorting();
    if (ret != SUCCESS) {
      GELOGE(ret, "[TopologicalSort][Graph]Graph name:%s.", graph->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "Failed to topological sort for graph:%s.", graph->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::GetBufferPoolAndPeerCalcNodes(const ComputeGraphPtr &graph) {
  std::unordered_map<std::string, std::unordered_map<int64_t, std::set<NodePtr>>> unique_calc_nodes;
  for (const NodePtr &node : graph->GetAllNodes()) {
    auto in_data_nodes = node->GetInAllNodes();
    for (NodePtr &in_node : in_data_nodes) {
      int64_t buffer_pool_id = 0;
      int64_t buffer_pool_size = 0;
      bool get_attr = AttrUtils::GetInt(in_node->GetOpDesc(), ATTR_NAME_BUFFER_POOL_ID, buffer_pool_id);
      get_attr = get_attr && (AttrUtils::GetInt(in_node->GetOpDesc(), ATTR_NAME_BUFFER_POOL_SIZE, buffer_pool_size));
      if (get_attr) {
        std::string batch_label;
        (void) AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label);
        peer_buffer_node_item_[batch_label][node].emplace_back(BufferPoolNodeItem(in_node, 0, 0));
        buffer_node_to_calc_[batch_label][in_node] = node;
        if (unique_calc_nodes[batch_label][buffer_pool_id].count(node) == 0) {
          calc_nodes_[batch_label][buffer_pool_id].emplace_back(node);
          unique_calc_nodes[batch_label][buffer_pool_id].insert(node);
        }
        GELOGI("[Get][BufferNode]Calc node:%s, pool node:%s.", node->GetName().c_str(), in_node->GetName().c_str());
        Status ret = SetBufferPoolSize(batch_label, buffer_pool_id, buffer_pool_size);
        if (ret != SUCCESS) {
          GELOGE(ret, "[Set][BufferPoolSize]Node:%s", in_node->GetName().c_str());
          REPORT_INNER_ERROR("E19999", "Failed to set buffer pool size, something wrong with the info of node:%s",
                             in_node->GetName().c_str());
          return ret;
        }
      }
    }
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::SetBufferPoolSize(const std::string &batch_label, int64_t id, int64_t size) {
  auto iter = buffer_pool_size_[batch_label].find(id);
  if (iter != buffer_pool_size_[batch_label].end() && iter->second != size) {
    GELOGE(PARAM_INVALID, "[Check][BufferPoolSize]Get different size with the same id, "
           "id:%ld, original size:%ld, this size:%ld.", id, iter->second, size);
    REPORT_INNER_ERROR("E19999", "Get different size with the same id, "
                       "id:%ld, original size:%ld, this size:%ld.", id, iter->second, size);
    return PARAM_INVALID;
  }
  buffer_pool_size_[batch_label][id] = size;
  return SUCCESS;
}

Status BufferPoolMemoryPass::AllocateAllBufferPoolSpace() {
  for (const auto &iter : calc_nodes_) {
    std::string batch_label = iter.first;
    Status ret = AllocateSpaceInBatch(calc_nodes_[batch_label],
                                      buffer_pool_size_[batch_label],
                                      buffer_node_to_calc_[batch_label],
                                      peer_buffer_node_item_[batch_label]);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Alloc][InBatch]Batch_label:%s.", batch_label.c_str());
      REPORT_INNER_ERROR("E19999", "Failed to allocate space in batch, batch_label:%s.", batch_label.c_str());
      return ret;
    }
    GELOGI("[Alloc][InBatch]Alloc space in batch successfully, batch label:%s.", batch_label.c_str());
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::AllocateSpaceInBatch(
    const std::map<int64_t, std::vector<NodePtr>> &calc_nodes,
    const std::unordered_map<int64_t, int64_t> &buffer_pool_size_map,
    const std::unordered_map<NodePtr, NodePtr> &buffer_node_to_calc,
    std::unordered_map<NodePtr, std::vector<BufferPoolNodeItem>> &buffer_pool_nodes_item) {
  for (const auto &calc_node_in_pool : calc_nodes) {
    int64_t pool_id = calc_node_in_pool.first;
    int64_t buffer_pool_size = buffer_pool_size_map.at(pool_id);
    ClearQueue(mem_ctrl_event_);
    ClearQueue(stream_ctrl_event_);
    BufferPool buffer_pool(pool_id, buffer_pool_size, buffer_node_to_calc);
    Status ret = AllocateSpaceInBufferPool(buffer_pool,
                                           calc_node_in_pool.second,
                                           buffer_pool_nodes_item);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Alloc][InBufferPool]Pool id:%ld, pool size:%ld.", pool_id, buffer_pool_size);
      REPORT_INNER_ERROR("E19999", "Failed to allocate space in buffer pool, id:%ld, pool size:%ld.",
                         pool_id, buffer_pool_size);
      return ret;
    }
    GELOGI("[Alloc][InBufferPool]Alloc space in buffer pool successfully, pool id:%ld.", pool_id);
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::AllocateSpaceInBufferPool(
    const BufferPool &buffer_pool,
    const std::vector<NodePtr> &calc_nodes_in_pool,
    std::unordered_map<NodePtr, std::vector<BufferPoolNodeItem>> &buffer_pool_nodes_item) {
  int64_t pool_id = buffer_pool.pool_id;
  int64_t buffer_pool_size = buffer_pool.pool_size;
  int64_t next_start = 0;
  NodePtr pre_buffer_pool_node = nullptr;
  std::queue<BufferPoolNodeItem> node_mem_range_in_pool;
  node_mem_range_in_pool.push(BufferPoolMemoryPass::BufferPoolNodeItem(nullptr, 0, buffer_pool_size));
  for (auto &calc_node : calc_nodes_in_pool) {
    auto &peer_buffer_node_item = buffer_pool_nodes_item[calc_node];
    std::unordered_map<int64_t, int64_t> calc_total_size;
    size_t input_buffer_node_num = 0;
    for (auto &node_item : peer_buffer_node_item) {
      auto peer_buffer_node = node_item.node;
      GE_CHECK_NOTNULL(peer_buffer_node);
      int64_t total_size = 0;
      ++input_buffer_node_num;
      Status ret = GetMemorySize(peer_buffer_node, total_size);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Get][MemSize]Node:%s, calc_node:%s.",
               peer_buffer_node->GetName().c_str(), calc_node->GetName().c_str());
        REPORT_INNER_ERROR("E19999", "Failed to get memory size, node:%s, calc_node:%s.",
                           peer_buffer_node->GetName().c_str(), calc_node->GetName().c_str());
        return ret;
      }
      ret = CheckBufferPoolSize(total_size, pool_id, buffer_pool_size, calc_total_size);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Check][BufferPoolSize]Capacity is not enough for all data, calc_node:%s.",
               calc_node->GetName().c_str());
        REPORT_INNER_ERROR("E19999", "Capacity is not enough for all data, calc_node:%s.",
                           calc_node->GetName().c_str());
        return ret;
      }
      BufferPoolNodeItem buffer_pool_node_item(peer_buffer_node, calc_node, pre_buffer_pool_node, total_size,
                                               0, 0, (input_buffer_node_num == peer_buffer_node_item.size()));
      ret = AllocateSpaceForBufferPoolNode(next_start, buffer_pool, buffer_pool_node_item, node_mem_range_in_pool);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Alloc][ForNode]Pool node:%s, calc_node:%s.",
               peer_buffer_node->GetName().c_str(), calc_node->GetName().c_str());
        REPORT_INNER_ERROR("E19999", "Failed to allocate space for buffer pool node:%s, calc_node:%s.",
                           peer_buffer_node->GetName().c_str(), calc_node->GetName().c_str());
        return ret;
      }
      pre_buffer_pool_node = peer_buffer_node;
    }
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::AllocateSpaceForBufferPoolNode(int64_t &next_start,
                                                            const BufferPool buffer_pool,
                                                            BufferPoolNodeItem &buffer_pool_node_item,
                                                            std::queue<BufferPoolNodeItem> &node_mem_range_in_pool) {
  // Get event id must be before FixTheTimingOfDependentNodes
  uint32_t logic_event = logic_event_num_;
  NodePtr buffer_node = buffer_pool_node_item.node;
  NodePtr calc_node = buffer_pool_node_item.out_calc_node;
  /// In the scenario where there are multiple PREFETCH operators in the inputs of the calculation operator,
  /// the addition of events is optimized to only add events after the last PREFETCH operator.
  ///              w1         w2         w3         w4         w5
  ///              |          |          |          |          |
  ///          prefetch1  prefetch2  prefetch3  prefetch4  prefetch5   xxx
  ///               \         /           \         /          \       /
  ///                \       /             \       /            \     /
  ///                 \     /               \     /              \   /
  ///                  node1                 node2               node3
  ///                   |                     |                   |
  ///                   |                     |                   |
  ///                    ---------------  other nodes  ------------
  ///
  /// The event id of the PREFETCH operator to the calculation operator needs to be generated before
  /// FixTheTimingOfDependentNodes, because FixTheTimingOfDependentNodes may add a new id to stream_ctrl_event_,
  /// and this id cannot be reused until the next PREFETCH operator in the sequence.
  if (buffer_pool_node_item.is_last_input) {
    logic_event = GenerateEventId(buffer_node->GetName(), stream_ctrl_event_);
    node_event_multiplexing_[buffer_node].push_back(string("SendTo;" + calc_node->GetName() +
                                                           ";" + std::to_string(logic_event)));
    mem_ctrl_event_.push(std::make_pair(calc_node->GetName(), logic_event));
  }
  NodePtr dependent_calc_node = GetOffsetAndDependency(next_start, buffer_pool_node_item.total_size,
                                                       buffer_pool.pool_size,
                                                       buffer_pool.buffer_node_to_calc,
                                                       node_mem_range_in_pool);
  if (dependent_calc_node != nullptr) {
    Status ret = FixTheTimingOfDependentNodes(dependent_calc_node, buffer_node);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Fix][Timing]Pool_id:%ld, pool node:%s, dependent node:%s.",
             buffer_pool.pool_id, buffer_node->GetName().c_str(), dependent_calc_node->GetName().c_str());
      REPORT_INNER_ERROR("E19999", "Failed to fix timing, pool_id:%ld, pool node:%s, dependent node:%s.",
                         buffer_pool.pool_id, buffer_node->GetName().c_str(),
                         dependent_calc_node->GetName().c_str());
      return ret;
    }
  }

  buffer_pool_node_item.offset_start = next_start;
  buffer_node_logical_offset_[buffer_node].push_back(buffer_pool_node_item.total_size);
  buffer_node_logical_offset_[buffer_node].push_back(next_start);
  FMK_INT64_ADDCHECK(next_start, buffer_pool_node_item.total_size);
  next_start += buffer_pool_node_item.total_size;
  buffer_pool_node_item.offset_end = next_start;
  node_mem_range_in_pool.push(buffer_pool_node_item);
  if (buffer_pool_node_item.pre_buffer_pool_node != nullptr) {
    bool not_change = true;
    auto ret = TryToFixNodeOrder(buffer_pool_node_item.pre_buffer_pool_node, buffer_node, not_change);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Fix][BufferPoolNodeOrder]Pre node:%s, curr node:%s.",
             buffer_pool_node_item.pre_buffer_pool_node->GetName().c_str(), buffer_node->GetName().c_str());
      return ret;
    }
  }
  GELOGI("[Alloc][ForNode]Buffer pool node %s send to %s, offset start:%ld, send event id:%u.",
         buffer_node->GetName().c_str(), calc_node->GetName().c_str(),
         buffer_pool_node_item.offset_start, logic_event);
  return SUCCESS;
}

/// When generating the event ID, determine whether the name of the queue head node is the same as the name of
/// the operator, in order to handle such scenarios:
///              w1         w2         w3        w4         w5
///              |          |          |         |          |
///          prefetch1  prefetch2  prefetch3  prefetch4  prefetch5
///             |          |          |         |          |
///           node1      node2      node3     node4      node5
///
///  Memory distribution:
///
///      |____w1_____|__|
///
///      |____w2_____|__|
///
///      |____w3_____|__|
///
///      |______w4______|
///
///      |______w5______|
///
/// In this scenario, prefetch2 depends on node1. If the dependency is handled by adding an event of node1 to prefetch2,
/// the id sent by prefetch2 will be the same as the id it receives.Although Runtime supports this through WaitReset,
/// we consider this a dangerous operation and avoid it.
uint32_t BufferPoolMemoryPass::GenerateEventId(const std::string &node_name,
                                               std::queue<std::pair<std::string, uint32_t>> &event_queue) {
  uint32_t logic_event = logic_event_num_;
  if (!event_queue.empty()) {
    auto item = event_queue.front();
    if (item.first != node_name) {
      logic_event = item.second;
      event_queue.pop();
      return logic_event;
    }
  }
  ++logic_event_num_;
  return logic_event;
}

NodePtr BufferPoolMemoryPass::GetOffsetAndDependency(int64_t &next_start,
    int64_t total_mem_size,
    int64_t buffer_pool_size,
    const std::unordered_map<NodePtr, NodePtr> &buffer_node_to_calc,
    std::queue<BufferPoolMemoryPass::BufferPoolNodeItem> &nodes_in_buffer) {
  // The buffer pool can no longer fit this Tensor and needs to turn back.
  if (next_start + total_mem_size > buffer_pool_size) {
    next_start = 0;
    if (!nodes_in_buffer.empty()) {
      // Take up the rest of the space at the end,
      nodes_in_buffer.back().offset_end = buffer_pool_size;
      // Pop the first tensor memory in the previous round of the previous round.
      nodes_in_buffer.pop();
    }
    while (!nodes_in_buffer.empty()) {
      auto node_item = nodes_in_buffer.front();
      // Go to the begin of previous round.
      if (node_item.offset_start == 0) {
        break;
      }
      nodes_in_buffer.pop();
    }
  }

  while (!nodes_in_buffer.empty()) {
    auto node_item = nodes_in_buffer.front();
    if (next_start + total_mem_size <= node_item.offset_end) {
      auto pool_node = node_item.node;
      if (pool_node == nullptr) {
        return nullptr;
      }
      auto output_calc = buffer_node_to_calc.find(pool_node);
      if (output_calc != buffer_node_to_calc.end()) {
        return output_calc->second;
      }
      return nullptr;
    }
    nodes_in_buffer.pop();
  }
  return nullptr;
}

Status BufferPoolMemoryPass::FixTheTimingOfDependentNodes(NodePtr &dependent_calc_node, NodePtr &curr_pool_node) {
  // The previous process ensures that all pointers are not null.
  bool not_change = false;
  Status ret = TryToFixNodeOrder(dependent_calc_node, curr_pool_node, not_change);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Fix][NodeOrder]Src:%s, dst:%s.",
           dependent_calc_node->GetName().c_str(), curr_pool_node->GetName().c_str());
    return ret;
  }
  if (not_change) {
    return SUCCESS;
  }
  uint32_t logic_event = GenerateEventId(dependent_calc_node->GetName(), mem_ctrl_event_);
  node_event_multiplexing_[curr_pool_node].push_back(string("RecvFrom;" + dependent_calc_node->GetName() +
                                                            ";" + std::to_string(logic_event)));
  stream_ctrl_event_.push(std::make_pair(curr_pool_node->GetName(), logic_event));
  GELOGI("[Fix][Timing]Add ctrl edge for buffer pool memory from %s to %s, buffer pool node recv event:%u.",
         dependent_calc_node->GetName().c_str(), curr_pool_node->GetName().c_str(), logic_event);
  return SUCCESS;
}

Status BufferPoolMemoryPass::SetResultOfMemoryAndEvent() {
  for (auto &iter : node_event_multiplexing_) {
    auto node = iter.first;
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    bool ret = AttrUtils::SetListStr(op_desc, ATTR_NAME_EVENT_MULTIPLEXING, iter.second);
    if (!ret) {
      GELOGE(INTERNAL_ERROR, "[Set][Attr]Node:%s.", node->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "Failed to set event reuse info, node:%s.", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    auto offset_iter = buffer_node_logical_offset_.find(node);
    if (offset_iter == buffer_node_logical_offset_.end()) {
      GELOGE(INTERNAL_ERROR, "[Get][LogicalOffset]Node:%s.", node->GetName().c_str());
      REPORT_INNER_ERROR("E19999", "Failed to get logical offset and size, node:%s.", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    ret = AttrUtils::SetListInt(op_desc, ATTR_NAME_BUFFER_POOL_NODE_SIZE_AND_OFFSET, offset_iter->second);
    if (!ret) {
      GELOGE(INTERNAL_ERROR, "[Set][Attr]Node:%s.", node->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "Failed to set node memory offset and size, node:%s.", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}
}  // namespace ge
