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

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_GREATER_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_GREATER_KERNEL_H_

#include <set>
#include <vector>

#include "common/fp16_t.h"
#include "graph/ge_tensor.h"
#include "inc/kernel.h"

namespace ge {
class GreaterKernel : public Kernel {
 public:
  Status Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  Status ComputeOutData(ConstGeTensorPtr input_x1, ConstGeTensorPtr input_x2, std::vector<int64_t> &x1_indexes,
                        std::vector<int64_t> &x2_indexes, std::vector<uint8_t> &y_data);

  Status GreaterCheck(const std::vector<ConstGeTensorPtr> &input);

  const std::set<DataType> greater_supported_type = {
      DT_FLOAT, DT_FLOAT16, DT_INT8,   DT_INT16,  DT_UINT16, DT_UINT8,
      DT_INT32, DT_INT64,   DT_UINT32, DT_UINT64, DT_BOOL,   DT_DOUBLE,
  };
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_GREATER_KERNEL_H_
