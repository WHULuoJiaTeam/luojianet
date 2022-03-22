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

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_MUL_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_MUL_KERNEL_H_

#include <vector>

#include "graph/ge_tensor.h"
#include "inc/kernel.h"
#include "common/fp16_t.h"

namespace ge {
class MulKernel : public Kernel {
 public:
  Status Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  Status MulCheck(const std::vector<ConstGeTensorPtr> &input);
  std::vector<int8_t> y_data_int8_t_;
  std::vector<int16_t> y_data_int16_t_;
  std::vector<int32_t> y_data_int32_t_;
  std::vector<int64_t> y_data_int64_t_;
  std::vector<uint8_t> y_data_uint8_t_;
  std::vector<uint16_t> y_data_uint16_t_;
  std::vector<uint32_t> y_data_uint32_t_;
  std::vector<uint64_t> y_data_uint64_t_;
  std::vector<fp16_t> y_data_fp16_t_;
  std::vector<float> y_data_float_;
  std::vector<double> y_data_double_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_MUL_KERNEL_H_
