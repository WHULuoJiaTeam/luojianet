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

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_GATHER_V2_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_GATHER_V2_KERNEL_H_

#include <vector>

#include "inc/kernel.h"

namespace ge {
class GatherV2Kernel : public Kernel {
 public:
  Status Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  template <typename T>
  Status ProcessAxis0(ConstGeTensorPtr tensor_x, GeTensorPtr output);
  template <typename T>
  Status ProcessAxis1(ConstGeTensorPtr tensor_x, GeTensorPtr output);
  template <typename T>
  Status ProcessAxis2(ConstGeTensorPtr tensor_x, GeTensorPtr output);
  template <typename T>
  Status ProcessAxis3(ConstGeTensorPtr tensor_x, GeTensorPtr output);
  template <typename T>
  Status GenData(const int64_t data_num, ConstGeTensorPtr tensor_x, int64_t axis, GeTensorPtr output);
  Status Check(const OpDescPtr &op_desc_ptr, const vector<ConstGeTensorPtr> &input,
               vector<GeTensorPtr> &v_output) const;
  Status CalcStride(std::vector<int64_t> &stride, std::vector<int64_t> dims);
  Status SaveIndicesByDataType(ConstGeTensorPtr indices_tensor_ptr, GeShape &x_shape, GeShape &indices_shape,
                               DataType indices_data_type, size_t axis);
  Status Process(int64_t axis, DataType data_type, ConstGeTensorPtr input_tensor_ptr, GeTensorPtr output_ptr);
  void DebugPrint(int64_t axis, const GeShape &x_shape, const GeShape &indices_shape,
                  const std::vector<int64_t> &y_shape);

 private:
  std::vector<int64_t> indicates_;
  std::vector<int64_t> xstride_;
  std::vector<int64_t> ystride_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_GATHER_V2_KERNEL_H_
