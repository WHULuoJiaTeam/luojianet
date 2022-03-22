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

#include "common/omg_util.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/math/math_util.h"

namespace ge {
///
/// @brief get the Original Type of FrameworkOp
/// @param [in] node
/// @param [out] type
/// @return Status
///
Status GetOriginalType(const ge::NodePtr &node, string &type) {
  GE_CHECK_NOTNULL(node);
  type = node->GetType();
  GE_IF_BOOL_EXEC(type != FRAMEWORKOP, return SUCCESS);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  bool ret = ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);
  if (!ret) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s fail from op:%s(%s)", ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE.c_str(),
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s fail from op:%s(%s)", ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE.c_str(),
           node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  GELOGD("Get FrameWorkOp original type [%s]", type.c_str());
  return SUCCESS;
}

///
/// @brief set op stream_label
/// @param [in] node
/// @param [in] label
/// @return Status
///
Status SetStreamLabel(const ge::NodePtr &node, const std::string &label) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);

  if (!AttrUtils::SetStr(tmp_desc, ge::ATTR_NAME_STREAM_LABEL, label)) {
    REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_STREAM_LABEL.c_str(),
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_STREAM_LABEL.c_str(), node->GetName().c_str(),
           node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op cycle_event flag
/// @param [in] node
/// @return Status
///
Status SetCycleEvent(const ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetBool(tmp_desc, ge::ATTR_NAME_STREAM_CYCLE_EVENT_FLAG, true)) {
    REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_STREAM_CYCLE_EVENT_FLAG.c_str(),
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_STREAM_CYCLE_EVENT_FLAG.c_str(),
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op active_label_list
/// @param [in] node
/// @param [in] active_label_list
/// @return Status
///
Status SetActiveLabelList(const ge::NodePtr &node, const std::vector<std::string> &active_label_list) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetListStr(tmp_desc, ge::ATTR_NAME_ACTIVE_LABEL_LIST, active_label_list)) {
    REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_ACTIVE_LABEL_LIST.c_str(),
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_ACTIVE_LABEL_LIST.c_str(), node->GetName().c_str(),
           node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op branch_label
/// @param [in] node
/// @param [in] branch_label
/// @return Status
///
Status SetSwitchBranchNodeLabel(const ge::NodePtr &node, const std::string &branch_label) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetStr(tmp_desc, ge::ATTR_NAME_SWITCH_BRANCH_NODE_LABEL, branch_label)) {
    REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_SWITCH_BRANCH_NODE_LABEL.c_str(),
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_SWITCH_BRANCH_NODE_LABEL.c_str(),
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op true_branch flag
/// @param [in] node
/// @param [in] value
/// @return Status
///
Status SetSwitchTrueBranchFlag(const ge::NodePtr &node, bool value) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetBool(tmp_desc, ge::ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, value)) {
    REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG.c_str(),
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG.c_str(),
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op original name
/// @param [in] node
/// @param [in] orig_name
/// @return Status
///
Status SetOriginalNodeName(const ge::NodePtr &node, const std::string &orig_name) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetStr(tmp_desc, ge::ATTR_NAME_ORIG_NODE_NAME, orig_name)) {
    REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_ORIG_NODE_NAME.c_str(),
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_ORIG_NODE_NAME.c_str(), node->GetName().c_str(),
           node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op cyclic_dependence flag
/// @param [in] node
/// @return Status
///
Status SetCyclicDependenceFlag(const ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetBool(tmp_desc, ge::ATTR_NAME_CYCLIC_DEPENDENCE_FLAG, true)) {
    REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_CYCLIC_DEPENDENCE_FLAG.c_str(),
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_CYCLIC_DEPENDENCE_FLAG.c_str(),
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op next_iteration name
/// @param [in] Merge Node
/// @param [in] NextIteration Node
/// @return Status
///
Status SetNextIteration(const NodePtr &node, const NodePtr &next) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(next);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  GE_CHECK_NOTNULL(next->GetOpDesc());

  const auto SetIterationName = [](const OpDescPtr &op_desc, const std::string &name) {
    if (!AttrUtils::SetStr(op_desc, ATTR_NAME_NEXT_ITERATION, name)) {
      REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_NEXT_ITERATION.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_NEXT_ITERATION.c_str(), op_desc->GetName().c_str(),
             op_desc->GetType().c_str());
      return FAILED;
    }
    return SUCCESS;
  };

  GE_CHK_STATUS_RET_NOLOG(SetIterationName(node->GetOpDesc(), next->GetName()));
  GE_CHK_STATUS_RET_NOLOG(SetIterationName(next->GetOpDesc(), node->GetName()));
  return SUCCESS;
}

///
/// @brief Align the memory
/// @param [in/out] memory size
/// @param [in] alinment
/// @return void
///
void AlignMemSize(int64_t &mem_size, int64_t align_size) {
  if (mem_size <= 0) {
    return;
  }
  mem_size = (mem_size + align_size - 1) / align_size * align_size;
}

///
/// @brief Get memory size from tensor desc
/// @param [in] node
/// @param [out] memory size
/// @return Status
///
Status GetMemorySize(const NodePtr &node, int64_t &output_size) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  auto output_op_desc = node->GetOpDesc()->GetOutputDescPtr(kBufferPoolNodeOutIndex);
  GE_CHECK_NOTNULL(output_op_desc);
  int64_t size = 0;
  auto ret = ge::TensorUtils::GetSize(*output_op_desc, size);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Get][Size]Node:%s.", node->GetName().c_str());
    REPORT_INNER_ERROR("E19999", "Failed to get output size, node:%s.", node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  FMK_INT64_ADDCHECK(size, kBufferPoolMemAlignSize);
  AlignMemSize(size, kBufferPoolMemAlignSize);
  // The HCOM operator requires an additional 512 bytes before and after
  FMK_INT64_ADDCHECK(size, (kBufferPoolMemAlignSize + kBufferPoolMemAlignSize));
  output_size = kBufferPoolMemAlignSize + size + kBufferPoolMemAlignSize;
  return SUCCESS;
}

///
/// @brief Check Is Unknown shape Tensor
/// @param [in] tensor_desc
/// @return true: Unknown / false: Known
///
bool IsUnknownShapeTensor(const GeTensorDesc &tensor_desc) {
  const static int kUnknowShape = -1;
  const static int kUnknowRank = -2;
  for (auto dim_size : tensor_desc.GetShape().GetDims()) {
    if (dim_size == kUnknowShape || dim_size == kUnknowRank) {
      return true;
    }
  }

  return false;
}

///
/// @brief Set Op _control_flow_group flag
/// @param [in] node
/// @param [in] group, condition group index of node.
/// @return
///
void SetControlFlowGroup(const NodePtr &node, int64_t group) {
  GE_RT_VOID_CHECK_NOTNULL(node);
  const auto &op_desc = node->GetOpDesc();
  GE_RT_VOID_CHECK_NOTNULL(op_desc);

  // op_desc as AttrHolderAdapter valid, Set attribute always success, just log for check.
  GELOGD("[%s] Set control flow group index: %ld", node->GetName().c_str(), group);
  if (!AttrUtils::SetInt(op_desc, ATTR_NAME_CONTROL_FLOW_GROUP, group)) {
    REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_CONTROL_FLOW_GROUP.c_str(),
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_CONTROL_FLOW_GROUP.c_str(), node->GetName().c_str(),
           node->GetType().c_str());
  }
}
}  // namespace ge
