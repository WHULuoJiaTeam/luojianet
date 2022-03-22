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

#ifndef _GRAPH_PASSES_FOLDING_KERNEL_UNITTEST_UTILS_H_
#define _GRAPH_PASSES_FOLDING_KERNEL_UNITTEST_UTILS_H_

#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <vector>

#include "common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator.h"
#include "graph/passes/constant_folding_pass.h"
#include "host_kernels/broadcast_args_kernel.h"
#include "inc/kernel_factory.h"
#include "shape_refiner.h"

namespace ge {
namespace test {

/// construct input with i_shape_dims and i_data
/// The kernel function depends on the output shape and type,
/// but the unit test cannot load the libopsproto.so,
/// that registers the operator's infer-shape function,
/// so you need to set the output shape and type yourself.
template <typename T>
static bool ConstructOpDesc(const vector<vector<int64_t>> &i_shape_dims, const vector<vector<int64_t>> &o_shape_dims,
                            const vector<vector<T>> &i_data, DataType in_dt, DataType out_dt,
                            vector<ConstGeTensorPtr> &inputs, shared_ptr<OpDesc> &op_desc_ptr) {
  for (size_t i = 0; i < i_shape_dims.size(); i++) {
    auto dims_vec = i_shape_dims.at(i);
    GeTensorDesc tensor_desc(GeShape(dims_vec), FORMAT_NCHW, in_dt);
    auto data_vec = i_data.at(i);
    ConstGeTensorPtr tensor =
        make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(T));
    inputs.push_back(tensor);

    op_desc_ptr->AddInputDesc(tensor_desc);
  }

  for (size_t i = 0; i < o_shape_dims.size(); i++) {
    op_desc_ptr->AddOutputDesc(GeTensorDesc(GeShape(o_shape_dims[i]), FORMAT_NCHW, out_dt));
  }
  return true;
}

template <typename IT, typename OT>
static bool ConstFoldingKernelCheckShapeAndOutput(string &op_type, const vector<vector<int64_t>> &i_shape_dims,
                                                  const vector<vector<IT>> &i_data, DataType in_dt,
                                                  const vector<vector<int64_t>> &o_shape_dims,
                                                  const vector<vector<OT>> &o_data, DataType out_dt) {
#ifndef CHECK_NOT_EQ_RETURN_FALSE
#define CHECK_NOT_EQ_RETURN_FALSE(lhd, rhd) \
  if (lhd != rhd) {                         \
    return false;                           \
  }
  vector<ConstGeTensorPtr> inputs;
  auto op_desc_ptr = make_shared<OpDesc>(op_type, op_type);
  // construct input with i_shape_dims and i_data
  int64_t err_flag = ConstructOpDesc(i_shape_dims, o_shape_dims, i_data, in_dt, out_dt, inputs, op_desc_ptr);
  CHECK_NOT_EQ_RETURN_FALSE(err_flag, true);

  // call kernel's compute func
  auto kernel = KernelFactory::Instance().Create(op_type);
  if (kernel == nullptr) {
    return false;
  }

  std::vector<GeTensorPtr> v_output;
  Status status = kernel->Compute(op_desc_ptr, inputs, v_output);
  CHECK_NOT_EQ_RETURN_FALSE(ge::SUCCESS, status);

  // check output data
  CHECK_NOT_EQ_RETURN_FALSE(o_data.size(), v_output.size());
  for (size_t i = 0; i < v_output.size(); i++) {
    CHECK_NOT_EQ_RETURN_FALSE(o_data[i].size() * sizeof(OT), v_output[i]->GetData().GetSize());
    GeTensorPtr output = v_output.at(i);
    OT *out_data = const_cast<OT *>(reinterpret_cast<const OT *>(output->GetData().data()));
    for (size_t j = 0; j < o_data[i].size(); j++) {
      CHECK_NOT_EQ_RETURN_FALSE(o_data[i][j], out_data[j]);
    }
  }

  // check data type
  CHECK_NOT_EQ_RETURN_FALSE(out_dt, v_output[0]->GetTensorDesc().GetDataType());

  // check shape
  CHECK_NOT_EQ_RETURN_FALSE(o_shape_dims.size(), v_output.size());
  for (size_t i = 0; i < v_output.size(); i++) {
    auto out_dims = v_output[i]->GetTensorDesc().GetShape().GetDims();
    CHECK_NOT_EQ_RETURN_FALSE(out_dims, o_shape_dims[i]);
  }
#endif
  return true;
}

}  // namespace test
}  // namespace ge
#endif  //_GRAPH_PASSES_FOLDING_KERNEL_UNITTEST_UTILS_H_
