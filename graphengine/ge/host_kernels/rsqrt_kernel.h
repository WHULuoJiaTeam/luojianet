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

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_RSQRT_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_RSQRT_KERNEL_H_

#include <vector>

#include "inc/kernel.h"
#include "common/fp16_t.h"

namespace ge {
class RsqrtKernel : public Kernel {
 public:
  Status Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                 std::vector<ge::GeTensorPtr> &v_output) override;
 private:
  template<typename T>
  Status RsqrtCompute(ConstGeTensorPtr &input_tensor_ptr, GeTensorPtr &output_tensor_ptr);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_RSQRT_KERNEL_H_
