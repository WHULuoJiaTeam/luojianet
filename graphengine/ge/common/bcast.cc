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

#include "common/bcast.h"

#include <vector>

#include "common/math_util.h"
#include "common/util.h"

using domi::Status;

namespace ge {
Status BCast::GenerateBcastInfo(const kVecInt &sx, const kVecInt &sy) {
  if (sx.size() == 0 && sy.size() == 0) {
    result_.push_back(1);
    x_reshape_.push_back(1);
    x_bcast_.push_back(1);
    y_reshape_.push_back(1);
    y_bcast_.push_back(1);
  } else {
    kVecInt x = sx;
    kVecInt y = sy;
    Reverse(x);
    Reverse(y);
    ExtendTensorDim(x, y);
    GE_RETURN_WITH_LOG_IF_ERROR(SetShapeDifferentInfo(x, y), "[Set][ShapeDifferentInfo] GenerateBcastInfo failed.");
  }
  ReverseAllIntermediateShapes();
  return domi::SUCCESS;
}

Status BCast::SetShapeDifferentInfo(const kVecInt &x, const kVecInt &y) {
  const int64_t n = x.size();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t x_i = x[i];
    GE_CHECK_GE(x_i, 0);
    const int64_t y_i = y[i];
    GE_CHECK_GE(y_i, 0);
    int64_t output_i = 0;
    int64_t x_bcast_i = 0;
    int64_t y_bcast_i = 0;

    if (x_i == y_i) {
      output_i = x_i;
      x_bcast_i = 1;
      y_bcast_i = 1;
      if (x_i == 1) {
        grad_x_reduce_idx_.push_back(n - 1 - i);
        grad_y_reduce_idx_.push_back(n - 1 - i);
      }
    } else if (x_i == 1) {
      output_i = y_i;
      x_bcast_i = y_i;
      y_bcast_i = 1;
      grad_x_reduce_idx_.push_back(n - 1 - i);
    } else if (y_i == 1) {
      output_i = x_i;
      x_bcast_i = 1;
      y_bcast_i = x_i;
      grad_y_reduce_idx_.push_back(n - 1 - i);
    } else {
      REPORT_INNER_ERROR("E19999", "SetShapeDifferentInfo failed. Two tensor shapes are not compatible "
                         "according to the broadcasting rule.");
      GELOGE(domi::PARAM_INVALID,
             "[Check][Param] SetShapeDifferentInfo failed. Two tensor shapes are not compatible "
             "according to the broadcasting rule.");
      return domi::PARAM_INVALID;
    }
    output_.push_back(output_i);
    result_.push_back(output_i);
    x_reshape_.push_back(x_i);
    x_bcast_.push_back(x_bcast_i);
    y_reshape_.push_back(y_i);
    y_bcast_.push_back(y_bcast_i);
  }
  return domi::SUCCESS;
}

void BCast::ExtendTensorDim(kVecInt &v_x, kVecInt &v_y) {
  if (v_x.size() > v_y.size()) {
    v_y.resize(v_x.size(), 1);
  } else {
    v_x.resize(v_y.size(), 1);
  }
}

BCast::kVecInt BCast::TransShapeToDimVec(const GeTensorDesc &shape) {
  const size_t dim_num = shape.GetShape().GetDimNum();
  BCast::kVecInt ret(dim_num);
  for (size_t i = 0; i < dim_num; ++i) {
    ret[i] = shape.GetShape().GetDim(i);
  }
  return ret;
}

void BCast::Reverse(kVecInt &shape) { std::reverse(shape.begin(), shape.end()); }

void BCast::ReverseAllIntermediateShapes() {
  // Reverse all intermediate shape params
  Reverse(x_reshape_);
  Reverse(x_bcast_);
  Reverse(y_reshape_);
  Reverse(y_bcast_);
  Reverse(result_);
  Reverse(output_);
  Reverse(grad_x_reduce_idx_);
  Reverse(grad_y_reduce_idx_);
}

void BCast::BCastIndexes(kVecInt &x_indexes, kVecInt &y_indexes) {
  Reverse(x_reshape_);
  Reverse(y_reshape_);
  Reverse(output_);

  // Process 0-th dimension
  int64_t x_dim = 1;
  int64_t y_dim = 1;
  int64_t out_dim = 1;

  // If x and y are both scalar, then output_ is empty
  if (!output_.empty()) {
    x_dim = x_reshape_.at(0);
    y_dim = y_reshape_.at(0);
    out_dim = output_.at(0);
  }

  int64_t x_bias = x_dim;
  int64_t y_bias = y_dim;

  for (int64_t i = 0; i < out_dim; i++) {
    x_indexes.push_back(x_dim == 1 ? 0 : i);
    y_indexes.push_back(y_dim == 1 ? 0 : i);
  }

  // Process the remaining dimensions
  for (size_t i = 1; i < output_.size(); i++) {
    x_dim = x_reshape_.at(i);  // i-th dimension of x.
    y_dim = y_reshape_.at(i);  // i-th dimension of y.
    out_dim = output_.at(i);   // i-th dimension of output_.

    int64_t stride = x_indexes.size();
    for (int64_t j = 1; j < out_dim; j++) {
      for (int64_t k = 0; k < stride; k++) {
        x_indexes.push_back(x_indexes.at(k) + (x_dim == 1 ? 0 : (j * x_bias)));
        y_indexes.push_back(y_indexes.at(k) + (y_dim == 1 ? 0 : (j * y_bias)));
      }
    }
    x_bias *= x_dim;
    y_bias *= y_dim;
  }

  Reverse(x_reshape_);
  Reverse(y_reshape_);
  Reverse(output_);
}
}  // namespace ge
