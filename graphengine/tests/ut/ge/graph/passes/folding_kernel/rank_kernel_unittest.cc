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
#include "host_kernels/rank_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/attr_value_util.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
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
using ge::SUCCESS;

class UtestGraphPassesFoldingKernelRankKernel : public testing::Test {
 protected:
  void SetUp() {
    graph = std::make_shared<ge::ComputeGraph>("default");
    op_desc_ptr = std::make_shared<OpDesc>("Rank", RANK);
    node = std::make_shared<Node>(op_desc_ptr, graph);
    kernel = KernelFactory::Instance().Create(RANK);
  }

  void TearDown() {}

  ge::ComputeGraphPtr graph;
  OpDescPtr op_desc_ptr;
  NodePtr node;
  shared_ptr<Kernel> kernel;
};

TEST_F(UtestGraphPassesFoldingKernelRankKernel, RankIsOne) {
  vector<int64_t> dims_vec_0 = {4};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc out_tensor_desc_0(GeShape(), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(tensor_desc_0);
  op_desc_ptr->AddOutputDesc(out_tensor_desc_0);
  std::vector<GeTensorPtr> v_output;
  Status status = kernel->Compute(node, v_output);

  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDimNum(), 0);
  EXPECT_EQ(*(int32_t *)(v_output[0]->GetData().data()), 1);
}

TEST_F(UtestGraphPassesFoldingKernelRankKernel, RankIsThree) {
  vector<int64_t> dims_vec_0 = {4, 2, 2};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc out_tensor_desc_0(GeShape(), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(tensor_desc_0);
  op_desc_ptr->AddOutputDesc(out_tensor_desc_0);
  std::vector<GeTensorPtr> v_output;
  Status status = kernel->Compute(node, v_output);

  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDimNum(), 0);
  EXPECT_EQ(*(int32_t *)(v_output[0]->GetData().data()), 3);
}

TEST_F(UtestGraphPassesFoldingKernelRankKernel, InvalidCaseInputSizeIsZero) {
  std::vector<GeTensorPtr> v_output;
  Status status = kernel->Compute(node, v_output);

  EXPECT_NE(SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelRankKernel, InvalidCaseInputSizeIsMoreThanOne) {
  vector<int64_t> dims_vec_0 = {4, 2, 2};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(tensor_desc_0);
  op_desc_ptr->AddInputDesc(tensor_desc_0);

  std::vector<GeTensorPtr> v_output;
  Status status = kernel->Compute(node, v_output);

  EXPECT_NE(SUCCESS, status);
}
