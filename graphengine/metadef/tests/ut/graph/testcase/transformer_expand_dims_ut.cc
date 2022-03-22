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
#define private public
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"
#include "expand_dimension.h"
#include "transfer_shape_according_to_format.h"
#include "transfer_range_according_to_format.h"

namespace transformer {
class TransformerExpandDimsUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(TransformerExpandDimsUT, not_expand_1) {
  ge::GeShape shape({8, 9});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, 0, "FORBIDDEN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {8, 9};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, not_expand_2) {
  ge::GeShape shape({8, 9});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ, 0, "", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {8, 9};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, not_expand_3) {
  ge::GeShape shape({8, 9});
  bool ret = ExpandDimension("Relu", ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, 0, "", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {8, 9};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, not_expand_4) {
  ge::GeShape shape({6, 7, 8, 9});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, 0, "", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {6, 7, 8, 9};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, not_expand_5) {
  ge::GeShape shape({5, 6, 7, 8, 9});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCDHW, ge::FORMAT_NC1HWC0, 0, "", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {5, 6, 7, 8, 9};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, default_reshape_type_1) {
  ge::GeShape shape;
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, 0, "HW", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 1, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, default_reshape_type_2) {
  ge::GeShape shape({5});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, 0, "", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 5, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, default_reshape_type_3) {
  ge::GeShape shape({7, 5});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, 0, "WC", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 7, 5, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, default_reshape_type_4) {
  ge::GeShape shape;
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, 0, "WC", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 1, 1, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, default_reshape_type_5) {
  ge::GeShape shape({7});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, 0, "WC", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 7, 1, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}
TEST_F(TransformerExpandDimsUT, default_reshape_type_6) {
  ge::GeShape shape({7, 5});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, 0, "WC", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 1, 1, 7, 5};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, default_reshape_type_7) {
  ge::GeShape shape({7, 5, 6});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, 0, "HWC", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 1, 7, 5, 6};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, default_reshape_type_8) {
  ge::GeShape shape({7, 5, 6, 4});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, 0, "HWC", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 7, 5, 6, 4};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, nhwc_reshape_type_1) {
  ge::GeShape shape;
  bool ret = ExpandDimension("Relu", ge::FORMAT_NHWC, ge::FORMAT_NDC1HWC0, 0, "", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 1, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, nhwc_reshape_type_2) {
  ge::GeShape shape({6});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NHWC, ge::FORMAT_NDC1HWC0, 0, "H", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 6, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, nhwc_reshape_type_3) {
  ge::GeShape shape({6, 7});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NHWC, ge::FORMAT_NDC1HWC0, 0, "NC", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {6, 1, 1, 7};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, nhwc_reshape_type_4) {
  ge::GeShape shape({5,6,7});
  bool ret = ExpandDimension("Relu", ge::FORMAT_NHWC, ge::FORMAT_NDC1HWC0, 0, "NWC", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {5, 1, 6, 7};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, cn_reshape_type_1) {
  ge::GeShape shape({5,6});
  bool ret = ExpandDimension("Relu", ge::FORMAT_HWCN, ge::FORMAT_NDC1HWC0, 0, "CN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 1, 5, 6};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, cn_reshape_type_2) {
  ge::GeShape shape({5,6});
  bool ret = ExpandDimension("Relu", ge::FORMAT_CHWN, ge::FORMAT_NDC1HWC0, 0, "CN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 1, 5, 6};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_1) {
  ge::GeShape shape;
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 1, 1, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_2) {
  ge::GeShape shape({7});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "W", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 1, 7, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_3) {
  ge::GeShape shape({7});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "WC", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 1, 7, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_4) {
  ge::GeShape shape({7});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "HWC", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 7, 1, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_5) {
  ge::GeShape shape({7, 5});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "HN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 7, 1, 1, 5};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_6) {
  ge::GeShape shape({7, 5});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "HWN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 7, 5, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_7) {
  ge::GeShape shape({7, 5});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "HCN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 7, 1, 5, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_8) {
  ge::GeShape shape({7, 5});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "DWN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {7, 1, 5, 1, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_9) {
  ge::GeShape shape({7, 5, 6});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "HCN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 7, 1, 5, 6};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_10) {
  ge::GeShape shape({7, 5, 6, 8});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "DHWC", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {7, 5, 6, 8, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_11) {
  ge::GeShape shape({7, 5, 6, 8});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "HWCN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 7, 5, 6, 8};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_12) {
  ge::GeShape shape({7, 5, 6, 8});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "DHCN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {7, 5, 1, 6, 8};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}
TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_13) {
  ge::GeShape shape({7, 5, 6, 8});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "DWCN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {7, 1, 5, 6, 8};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_14) {
  ge::GeShape shape({7, 5, 6, 8});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "DHWN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {7, 5, 6, 1, 8};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerExpandDimsUT, dhwcn_reshape_type_15) {
  ge::GeShape shape({7, 5, 6, 8});
  bool ret = ExpandDimension("Relu", ge::FORMAT_DHWCN, ge::FORMAT_NDC1HWC0, 0, "DHWCN", shape);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {7, 5, 6, 8, 1};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

}  // namespace ge