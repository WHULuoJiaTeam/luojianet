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

#include "graph/passes/assert_pass.h"

#include <map>
#include <queue>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"

namespace ge {
// aicpu not support string type, so current implemention is Upward traversal
Status AssertPass::Run(NodePtr &node) {
  GELOGD("AssertPass running");
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "param [node] must not be null.");
    return PARAM_INVALID;
  }
  if (node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param op_desc of node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] param [node] [opDesc] must not be null.");
    return PARAM_INVALID;
  }
  std::string op_type = node->GetOpDesc()->GetType();
  if (op_type == ASSERT) {
    GELOGD("op type is assert.");

    std::vector<NodePtr> nodes_unused;
    // collect assert and other unused ops
    CollectUnusedNode(node, nodes_unused);
    // remove unused node
    Status status = RemoveUnusedNode(nodes_unused);
    if (status != SUCCESS) {
      GELOGE(status, "[Remove][UnusedNode] failed, ret:%d.", status);
      return status;
    }
  }
  return SUCCESS;
}

void AssertPass::CollectUnusedNode(const NodePtr &assert_node, vector<NodePtr> &nodes_unused) {
  std::map<Node *, uint32_t> invalid_outdata_info;
  std::queue<NodePtr> node_queue;
  node_queue.push(assert_node);

  while (!node_queue.empty()) {
    NodePtr cur_node = node_queue.front();
    if (cur_node == nullptr) {
      continue;
    }
    node_queue.pop();
    nodes_unused.push_back(cur_node);

    for (const auto &src_node : cur_node->GetInDataNodes()) {
      if (src_node != nullptr && src_node->GetOpDesc() != nullptr) {
        auto size = ++invalid_outdata_info[src_node.get()];
        // src_node need to be deleted
        if (src_node->GetOutDataNodesSize() == size && src_node->GetOpDesc()->GetType() != DATA &&
            src_node->GetOpDesc()->GetType() != AIPPDATA) {
          node_queue.push(src_node);
        }
      }
    }
  }
}

Status AssertPass::RemoveUnusedNode(std::vector<NodePtr> &nodes_unused) {
  for (NodePtr &node : nodes_unused) {
    if (node == nullptr) {
      continue;
    }
    std::vector<int> assert_io_map;
    size_t out_nums = node->GetAllOutDataAnchorsSize();
    while (out_nums > 0) {
      assert_io_map.push_back(-1);
      out_nums--;
    }

    if (IsolateAndDeleteNode(node, assert_io_map) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Isolate and delete node:%s(%s) failed",
                        node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Call][IsolateAndDeleteNode] for node:%s(%s) failed",
             node->GetName().c_str(), node->GetType().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}
}  // namespace ge
