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

#include "graph/passes/guarantee_const_pass.h"

#include <string>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "common/omg_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
const uint32_t kGuaranteeConstInputsSize = 1;
}
Status GuaranteeConstPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  string type;
  Status status_ret = GetOriginalType(node, type);
  if (status_ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get original type for node:%s failed", node->GetName().c_str());
    GELOGE(status_ret, "[Get][OriginalType] for node:%s failed", node->GetName().c_str());
    return status_ret;
  }
  if (type != GUARANTEECONST) {
    return SUCCESS;
  }
  if (node->GetOpDesc()->GetAllInputsDesc().size() != kGuaranteeConstInputsSize) {
    REPORT_CALL_ERROR("E19999", "Num:%zu of input desc in node:%s(%s) not equal to %u, "
                      "check invalid", node->GetOpDesc()->GetAllInputsDesc().size(),
                      node->GetName().c_str(), node->GetType().c_str(), kGuaranteeConstInputsSize);
    GELOGE(PARAM_INVALID, "[Check][Param] Num:%zu of input desc in node:%s(%s) not equal to %u",
           node->GetOpDesc()->GetAllInputsDesc().size(),
           node->GetName().c_str(), node->GetType().c_str(), kGuaranteeConstInputsSize);
    return PARAM_INVALID;
  }
  // [Cascade pointer]
  const auto &in_desc = node->GetOpDesc()->MutableInputDesc(0);
  GE_CHECK_NOTNULL(in_desc);
  // Input tensor cannot be a resource variable handle.
  const DataType &input_dtype = in_desc->GetDataType();
  if (input_dtype == DT_RESOURCE) {
    REPORT_CALL_ERROR("E19999", "Data type:%s of op:%s(%s) input0 tensor not equal to %s, check invalid",
                      TypeUtils::DataTypeToSerialString(input_dtype).c_str(),
                      node->GetName().c_str(), node->GetType().c_str(),
                      TypeUtils::DataTypeToSerialString(DT_RESOURCE).c_str());
    GELOGE(FAILED, "[Check][Param] Data type:%s of op:%s(%s) input0 tensor not equal to %s",
           TypeUtils::DataTypeToSerialString(input_dtype).c_str(),
           node->GetName().c_str(), node->GetType().c_str(), TypeUtils::DataTypeToSerialString(DT_RESOURCE).c_str());
    return FAILED;
  }

  return IsolateAndDeleteNode(node, {0});
}
}  // namespace ge
