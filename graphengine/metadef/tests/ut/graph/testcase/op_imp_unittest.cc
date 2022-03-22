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
#include "graph/operator_reg.h"
#include <gtest/gtest.h>
#include <vector>

namespace ge {
class BroadCastInferUt : public testing::Test {};

TEST_F(BroadCastInferUt, Scalar1) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({1, 2, 3});},
      []() { return std::vector<int64_t>({}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 3}));

  ret_shape.clear();
  ret = BroadCastInfer(
      []() { return std::vector<int64_t>({});},
      []() { return std::vector<int64_t>({1, 2, 3}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 3}));
}

TEST_F(BroadCastInferUt, SameShape) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({1, 2, 3});},
      []() { return std::vector<int64_t>({1, 2, 3}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 3}));
}

TEST_F(BroadCastInferUt, BroadCastDim1) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({3, 2, 1});},
      []() { return std::vector<int64_t>({1, 2, 3}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({3, 2, 3}));
}

TEST_F(BroadCastInferUt, BroadCastRank) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({1, 2, 3, 4});},
      []() { return std::vector<int64_t>({3, 4}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 3, 4}));
}

TEST_F(BroadCastInferUt, BroadCastRankAndDim1) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({1, 2, 1, 4});},
      []() { return std::vector<int64_t>({5, 4}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 5, 4}));
}

TEST_F(BroadCastInferUt, BroadCastFailed_DimDiff) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({1, 2, 3, 4});},
      []() { return std::vector<int64_t>({5, 4}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(BroadCastInferUt, BroadCastRankAndDim1_1) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({5, 4});},
      []() { return std::vector<int64_t>({1, 2, 1, 4}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 5, 4}));
}
}  // namespace ge