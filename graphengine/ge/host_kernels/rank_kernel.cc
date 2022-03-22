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

#include "host_kernels/rank_kernel.h"

#include <memory>
#include <vector>

#include "external/graph/types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "inc/kernel_factory.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/common/types.h"

namespace {
const size_t kRankInputSize = 1;
const uint32_t kRankDataInputIndex = 0;
}  // namespace

namespace ge {
Status RankKernel::Compute(const NodePtr &node, std::vector<GeTensorPtr> &v_output) {
  if (node == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return FAILED;
  }
  OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  size_t input_node_size = op_desc->GetInputsSize();
  if (input_node_size != kRankInputSize) {
    GELOGW("input node size must be %zu", kRankInputSize);
    return NOT_CHANGED;
  }

  const auto &input_shape = op_desc->MutableInputDesc(kRankDataInputIndex);
  GE_CHECK_NOTNULL(input_shape);
  if (input_shape->GetShape().GetDims() == UNKNOWN_RANK) {
    return NOT_CHANGED;
  }
  auto ndims = input_shape->GetShape().GetDimNum();
  GeTensorDesc tensor_desc(op_desc->GetOutputDesc(0));
  GeTensorPtr output_ptr;
  output_ptr = MakeShared<ge::GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(&ndims), GetSizeByDataType(DT_INT32));
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "make_shared ge::GeTensor failed");
    return MEMALLOC_FAILED;
  }
  v_output.push_back(output_ptr);
  return SUCCESS;
}

REGISTER_KERNEL(RANK, RankKernel);
}  // namespace ge
