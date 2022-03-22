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

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_FLOORDIV_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_FLOORDIV_KERNEL_H_

#include <vector>

#include "inc/kernel.h"

namespace ge {
class FloorDivKernel : public Kernel {
 public:
  Status Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  Status FloorDivCheck(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input) const;
  void ShapeCal(const std::vector<ConstGeTensorPtr> &input, GeTensorPtr output_ptr);
  template <typename T>
  T DivCal(const T &x_i, const T &y_i);
  template <typename T>
  bool ZeroCheck(const T &element, DataType data_type);
  template <typename T>
  Status DataCalBroadcast(const T &x, const T &y, size_t num_x, size_t num_y, DataType data_type,
                          GeTensorPtr output_ptr);
  template <typename T>
  Status DataCal(const std::vector<ConstGeTensorPtr> &input, ge::GeTensorPtr output_ptr);
  Status ComputeByDataType(DataType data_type, const std::vector<ConstGeTensorPtr> &input, GeTensorPtr output_ptr);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_FLOORDIV_KERNEL_H_
