/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "graph/runtime_inference_context.h"
#undef private
#undef protected

namespace ge {
class RuntimeInferenceContextTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};


TEST_F(RuntimeInferenceContextTest, TestSetGetTensor) {
  RuntimeInferenceContext ctx;
  GeTensorDesc desc;
  GeTensorPtr ge_tensor = std::make_shared<GeTensor>(desc);
  ASSERT_EQ(ctx.SetTensor(1, 3, ge_tensor), GRAPH_SUCCESS);
  GeTensorPtr new_tensor;
  ASSERT_EQ(ctx.GetTensor(1, 3, new_tensor), GRAPH_SUCCESS);
  ASSERT_NE(ctx.GetTensor(2, 0, new_tensor), GRAPH_SUCCESS);
  ASSERT_NE(ctx.GetTensor(2, -1, new_tensor), GRAPH_SUCCESS);
  ASSERT_NE(ctx.GetTensor(1, 4, new_tensor), GRAPH_SUCCESS);
  ASSERT_NE(ctx.GetTensor(1, 0, new_tensor), GRAPH_SUCCESS);
  ctx.Release();
  ASSERT_NE(ctx.GetTensor(1, 3, new_tensor), GRAPH_SUCCESS);
}
} // namespace ge
