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

#include "graph/passes/permute_pass.h"
#include <queue>
#include <vector>
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "inc/kernel.h"
#include "inc/kernel_factory.h"
#include "framework/omg/omg_inner_types.h"
#include "common/local_context.h"

using domi::DOMI_TENSOR_ND;
using domi::DOMI_TENSOR_NHWC;
using domi::SUCCESS;
using domi::TENSORFLOW;

namespace ge {
Status PermutePass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  std::vector<NodePtr> isolate_nodes;
  for (NodePtr &node : graph->GetDirectNode()) {
    OpDescPtr op_desc_ptr = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc_ptr);
    GE_IF_BOOL_EXEC(
      op_desc_ptr->GetType() == PERMUTE && GetLocalOmgContext().type == domi::TENSORFLOW,
      /// Input format 5D means NHWC in 4D way. So if input origin foramt is NCHW and
      /// permute paramter list is [0,3,1,2], this permute can be optimised.
      GE_IF_BOOL_EXEC(
          GetLocalOmgContext().format != DOMI_TENSOR_ND,
        // Get input origin foramt
        for (NodePtr &n
             : graph->GetDirectNode()) {
          GE_IF_BOOL_EXEC(
            n->GetOpDesc()->GetType() == PERMUTE, std::queue<NodePtr> q_node; q_node.push(n); bool jump_out = false;
            while (!q_node.empty()) {
              NodePtr n_temp = q_node.back();
              q_node.pop();
              for (auto &inNode : n_temp->GetInDataNodes()) {
                int64_t cur_format = 0;
                GE_IF_BOOL_EXEC(AttrUtils::GetInt(inNode->GetOpDesc(), ATTR_NAME_FORMAT, cur_format),
                                GE_IF_BOOL_EXEC(!AttrUtils::SetInt(n->GetOpDesc(), "permute_src_format", cur_format),
                                                GELOGW("set permute_src_format failed");
                                                continue);
                                jump_out = true; break);
                q_node.push(inNode);
              }
              GE_IF_BOOL_EXEC(jump_out, break);
            });
        }

        int64_t permute_src_format = 0;
        GE_IF_BOOL_EXEC(!AttrUtils::GetInt(op_desc_ptr, "permute_src_format", permute_src_format), continue);
        // Get dim_index_
        std::vector<int64_t> index_list;
        GE_CHK_BOOL_RET_STATUS(AttrUtils::GetListInt(op_desc_ptr, PERMUTE_ATTR_ORDER, index_list), INTERNAL_ERROR,
                               "[Get][Attr] %s from op:%s(%s) failed", PERMUTE_ATTR_ORDER.c_str(),
                               op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());

        size_t index_size = index_list.size(); GE_IF_BOOL_EXEC(index_size == 0, continue);

        GE_IF_BOOL_EXEC(index_size == 4 && (permute_src_format == DOMI_TENSOR_NHWC && index_list.at(0) == 0 &&
                                            index_list.at(1) == 3 && index_list.at(2) == 1 && index_list.at(3) == 2),
                        isolate_nodes.push_back(node);
                        continue);
        int64_t conv_format = 0; GE_IF_BOOL_EXEC(
          index_size == 4 &&
            (index_list.at(0) == 0 && index_list.at(1) == 2 && index_list.at(2) == 3 && index_list.at(3) == 1),
          GE_IF_BOOL_EXEC(
            (node->GetOutDataNodesSize() > 0 && node->GetOutDataNodes().at(0) != nullptr &&
             node->GetOutDataNodes().at(0)->GetOpDesc() != nullptr) &&
              ((node->GetOutDataNodesSize() != 0 &&
                CONVOLUTION == node->GetOutDataNodes().at(0)->GetOpDesc()->GetType() &&
                AttrUtils::GetInt(node->GetOutDataNodes().at(0)->GetOpDesc(), ATTR_NAME_FORMAT, conv_format) &&
                conv_format == DOMI_TENSOR_NHWC) ||
               (node->GetOutDataNodesSize() != 0 &&
                node->GetOutDataNodes().at(0)->GetOpDesc()->GetType() == DEPCONVOLUTION) ||
               (node->GetOutDataNodesSize() != 0 &&
                node->GetOutDataNodes().at(0)->GetOpDesc()->GetType() == DECONVOLUTION) ||
               (node->GetOutDataNodesSize() != 0 && node->GetOutDataNodes().at(0)->GetOpDesc()->GetType() == PAD &&
                node->GetOutDataNodes().at(0)->GetOutDataNodesSize() != 0 &&
                node->GetOutDataNodes().at(0)->GetOutDataNodes().at(0) != nullptr &&
                node->GetOutDataNodes().at(0)->GetOutDataNodes().at(0)->GetOpDesc() != nullptr &&
                node->GetOutDataNodes().at(0)->GetOutDataNodes().at(0)->GetOpDesc()->GetType() == CONVOLUTION)),
            isolate_nodes.push_back(node);
            continue););););
  }

  GE_IF_BOOL_EXEC(
    isolate_nodes.size() != 0, for (auto &node
                                    : isolate_nodes) {
      // Adding an attribute indicates that the predecessor Permute has been deleted for the Builder to process.
      for (auto &outNode : node->GetOutDataNodes()) {
        OpDescPtr op_desc_ptr = outNode->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc_ptr);
        if (!AttrUtils::SetBool(op_desc_ptr, ATTR_NAME_PRED_PERMUTE_DELETED, true)) {
          REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_PRED_PERMUTE_DELETED.c_str(),
                            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
          GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_PRED_PERMUTE_DELETED.c_str(),
                 op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
          return INTERNAL_ERROR;
        }
      }
      GE_RETURN_WITH_LOG_IF_ERROR(graph->RemoveNode(node), "[Remove][Node] [%s] from graph:%s failed",
                                  node->GetOpDesc()->GetName().c_str(), graph->GetName().c_str());
    });
  return SUCCESS;
}
}  // namespace ge
