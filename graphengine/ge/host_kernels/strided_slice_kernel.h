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

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_STRIDED_SLICE_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_STRIDED_SLICE_KERNEL_H_

#include "inc/kernel.h"
#include <vector>

namespace ge {
class StridedSliceKernel : public Kernel {
 public:
  Status Compute(const OpDescPtr attr, const std::vector<ConstGeTensorPtr> &input,
                 vector<GeTensorPtr> &v_output) override;

 private:
  Status CheckAndGetAttr(const OpDescPtr &attr);
  static Status CheckInputParam(const std::vector<ConstGeTensorPtr> &input) ;
  Status InitParamWithAttrs(const std::vector<ConstGeTensorPtr> &input, std::vector<int64_t> &input_dims,
                            std::vector<int64_t> &begin_vec, std::vector<int64_t> &output_dims,
                            std::vector<int64_t> &stride_vec);
  Status MaskCal(const size_t i, int64_t &begin_i, int64_t &end_i, int64_t &dim_i) const;
  static Status StrideCal(const int64_t x_dims_i, int64_t &begin_i, int64_t &end_i, int64_t &stride_i,
                   int64_t &dim_final) ;
  void ExpandDimsWithNewAxis(const ConstGeTensorPtr &begin_tensor, const size_t x_dims_num, vector<int64_t> &x_dims);
  void ExpandStrideWithEllipsisMask(const size_t x_dims_num,
                                    const vector<int64_t> &x_dims, vector<int64_t> &orig_begin_vec,
                                    vector<int64_t> &orig_end_vec, vector<int64_t> &orig_stride_vec);

  void GetOutputDims(uint32_t dims_size, const std::vector<int64_t> &output_dims, vector<int64_t> &v_dims);

  map<string, uint32_t> attr_value_map_ = {{STRIDE_SLICE_ATTR_BEGIN_MASK, 0},
                                           {STRIDE_SLICE_ATTR_END_MASK, 0},
                                           {STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0},
                                           {STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0},
                                           {STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0}};
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_STRIDED_SLICE_KERNEL_H_
