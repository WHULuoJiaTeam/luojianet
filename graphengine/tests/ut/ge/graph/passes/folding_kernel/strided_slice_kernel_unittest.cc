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

#include <gtest/gtest.h>

#define protected public
#define private public
#include "host_kernels/strided_slice_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/attr_value_util.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/dimension_compute_pass.h"
#include "host_kernels/kernel_utils.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;

class UtestGraphPassesFoldingKernelStridedSliceKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CheckInputSize) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test2) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test3) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test4) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test5) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test6) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test7) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test8) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test9) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 = nullptr;

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test10) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 = nullptr;

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);

  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test11) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, DT_FLOAT16);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT16);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test12) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test13) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test14) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test15) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test16) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test17) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {10, 10, 10, 10};
  vector<int32_t> data_vec_0 = {3, 3, 3, 3};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
}
