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
class TransformerTransferShapeUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(TransformerTransferShapeUT, nchw_hwcn) {
  ge::GeShape shape({5, 6, 8, 9});
  ge::DataType dtype = DT_FLOAT16;
  ShapeAndFormat shape_and_format_info {shape, ge::FORMAT_NCHW, ge::FORMAT_HWCN, dtype};
  ShapeTransferAccordingToFormat shape_transfer;
  bool ret = shape_transfer.GetShapeAccordingToFormat(shape_and_format_info);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {8, 9, 6, 5};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerTransferShapeUT, nchw_nc1hwc0) {
  ge::GeShape shape({5, 18, 8, 9});
  ge::DataType dtype = DT_FLOAT16;
  ShapeAndFormat shape_and_format_info {shape, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, dtype};
  ShapeTransferAccordingToFormat shape_transfer;
  bool ret = shape_transfer.GetShapeAccordingToFormat(shape_and_format_info);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {5, 2, 8, 9, 16};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerTransferShapeUT, nhwc_fz) {
  ge::GeShape shape({8, 5, 6, 19});
  ge::DataType dtype = DT_FLOAT16;
  ShapeAndFormat shape_and_format_info {shape, ge::FORMAT_NHWC, ge::FORMAT_FRACTAL_Z, dtype};
  ShapeTransferAccordingToFormat shape_transfer;
  bool ret = shape_transfer.GetShapeAccordingToFormat(shape_and_format_info);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {60, 1, 16, 16};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerTransferShapeUT, nhwc_fz_with_group) {
  ge::GeShape shape({64, 5, 6, 19});
  ge::DataType dtype = DT_FLOAT16;
  ge::Format target_format = static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z), 20));
  ShapeAndFormat shape_and_format_info {shape, ge::FORMAT_NHWC, target_format, dtype};
  ShapeTransferAccordingToFormat shape_transfer;
  bool ret = shape_transfer.GetShapeAccordingToFormat(shape_and_format_info);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1140, 3, 16, 16};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerTransferShapeUT, nd_nz) {
  ge::GeShape shape({1, 18, 34});
  ge::DataType dtype = DT_FLOAT16;
  ShapeAndFormat shape_and_format_info {shape, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, dtype};
  ShapeTransferAccordingToFormat shape_transfer;
  bool ret = shape_transfer.GetShapeAccordingToFormat(shape_and_format_info);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {1, 3, 2, 16, 16};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

TEST_F(TransformerTransferShapeUT, nd_fz) {
  ge::GeShape shape({18, 34});
  ge::DataType dtype = DT_FLOAT16;
  ShapeAndFormat shape_and_format_info {shape, ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, dtype};
  ShapeTransferAccordingToFormat shape_transfer;
  bool ret = shape_transfer.GetShapeAccordingToFormat(shape_and_format_info);
  ASSERT_EQ(ret, true);
  std::vector<int64_t> expect_vec = {2, 3, 16, 16};
  ASSERT_EQ(shape.GetDims(), expect_vec);
}

}  // namespace ge