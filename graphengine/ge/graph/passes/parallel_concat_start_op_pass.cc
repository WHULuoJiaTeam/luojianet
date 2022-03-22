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

#include "graph/passes/parallel_concat_start_op_pass.h"
#include <vector>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

namespace ge {
namespace {
const size_t kParallelConcatStartOutputSize = 1;
const uint32_t kParallelConcatStartOutputDataIndex = 0;
const char *const kAttrDtype = "dtype";
const char *const kAttrShape = "shape";
}  // namespace
Status ParallelConcatStartOpPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  if (node->GetType() != PARALLELCONCATSTART) {
    return SUCCESS;
  }

  OpDescPtr node_op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(node_op_desc);
  string node_name = node->GetName();
  GELOGI("Start to replace operator _ParallelConcatStart with Constant, node name: %s.", node_name.c_str());

  if (node_op_desc->GetOutputsSize() != kParallelConcatStartOutputSize) {
    REPORT_INNER_ERROR("E19999", "Output tensor num:%zu of node:%s(%s) != %zu, check invalid",
                       node_op_desc->GetOutputsSize(), node_op_desc->GetName().c_str(),
                       node_op_desc->GetType().c_str(), kParallelConcatStartOutputSize);
    GELOGE(PARAM_INVALID, "[Check][Param] Node[%s] output size is unexpected, the value is %zu, expected valude:%zu.",
           node_name.c_str(), node_op_desc->GetOutputsSize(), kParallelConcatStartOutputSize);
    return PARAM_INVALID;
  }
  auto output_tensor_desc = node_op_desc->GetOutputDesc(kParallelConcatStartOutputDataIndex);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "New GeTensor failed");
    GELOGE(MEMALLOC_FAILED, "[New][GeTensor] failed");
    return FAILED;
  }

  ge::DataType attr_dtype;
  if (!ge::AttrUtils::GetDataType(node_op_desc, kAttrDtype, attr_dtype)) {
    REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", kAttrDtype,
                      node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Get][Attr] %s from op:%s(%s) failed", kAttrDtype,
           node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  output_ptr->MutableTensorDesc().SetDataType(attr_dtype);

  vector<int64_t> attr_shape_list;
  if (!ge::AttrUtils::GetListInt(node_op_desc, kAttrShape, attr_shape_list)) {
    REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", kAttrShape,
                      node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Get][Attr] %s from op:%s(%s) failed", kAttrShape,
           node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  output_ptr->MutableTensorDesc().SetShape(GeShape(attr_shape_list));

  vector<GeTensorPtr> outputs;
  outputs.emplace_back(output_ptr);

  return Folding(node, outputs);
}
}  // namespace ge
