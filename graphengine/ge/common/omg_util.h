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

#ifndef GE_GRAPH_COMMON_OMG_UTIL_H_
#define GE_GRAPH_COMMON_OMG_UTIL_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/node.h"

namespace ge {
static constexpr int64_t kBufferPoolMemAlignSize = 512;
static constexpr uint32_t kBufferPoolNodeOutIndex = 0;
static constexpr uint32_t kEventReuseThreshold = 65500;

///
/// @brief get the Original Type of FrameworkOp
/// @param [in] node
/// @param [out] type
/// @return Status
///
Status GetOriginalType(const ge::NodePtr &node, string &type);

///
/// @brief set op stream_label
/// @param [in] node
/// @param [in] label
/// @return Status
///
Status SetStreamLabel(const ge::NodePtr &node, const std::string &label);

///
/// @brief set op cycle_event flag
/// @param [in] node
/// @return Status
///
Status SetCycleEvent(const ge::NodePtr &node);

///
/// @brief set op active_label_list
/// @param [in] node
/// @param [in] label
/// @return Status
///
Status SetActiveLabelList(const ge::NodePtr &node, const std::vector<std::string> &active_label_list);

///
/// @brief set op branch_label
/// @param [in] node
/// @param [in] branch_label
/// @return Status
///
Status SetSwitchBranchNodeLabel(const ge::NodePtr &node, const std::string &branch_label);

///
/// @brief set op true_branch flag
/// @param [in] node
/// @param [in] value
/// @return Status
///
Status SetSwitchTrueBranchFlag(const ge::NodePtr &node, bool value);

///
/// @brief set op original name
/// @param [in] node
/// @param [in] orig_name
/// @return Status
///
Status SetOriginalNodeName(const ge::NodePtr &node, const std::string &orig_name);

///
/// @brief set op cyclic_dependence flag
/// @param [in] node
/// @return Status
///
Status SetCyclicDependenceFlag(const ge::NodePtr &node);

///
/// @brief set op next_iteration name
/// @param [in] Merge Node
/// @param [in] NextIteration Node
/// @return Status
///
Status SetNextIteration(const NodePtr &node, const NodePtr &next);

///
/// @brief Align the memory
/// @param [in/out] memory size
/// @param [in] alinment
/// @return void
///
void AlignMemSize(int64_t &mem_size, int64_t align_size);

///
/// @brief Get memory size from tensor desc
/// @param [in] node
/// @param [out] memory size
/// @return Status
///
Status GetMemorySize(const NodePtr &node, int64_t &output_size);

///
/// @brief Check Is Unknown shape Tensor
/// @param [in] tensor_desc
/// @return true: Unknown / false: Known
///
bool IsUnknownShapeTensor(const GeTensorDesc &tensor_desc);

///
/// @brief Set Op _control_flow_group flag
/// @param [in] node
/// @param [in] group, condition group index of node.
/// @return
///
void SetControlFlowGroup(const NodePtr &node, int64_t group);
}  // namespace ge

#endif  // GE_GRAPH_COMMON_OMG_UTIL_H_
