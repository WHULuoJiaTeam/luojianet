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

#include "common/ge_format_util.h"

#include "common/formats/formats.h"

namespace ge {
TEST(UtestGeFormatUtilTest, test_trans_shape_failure) {
  Shape shape({1, 3, 224, 224});
  TensorDesc src_desc(shape, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> dst_shape;
  EXPECT_NE(GeFormatUtil::TransShape(src_desc, FORMAT_RESERVED, dst_shape), SUCCESS);
}

TEST(UtestGeFormatUtilTest, test_trans_shape_success) {
  Shape shape({1, 3, 224, 224});
  TensorDesc src_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  std::vector<int64_t> dst_shape;
  std::vector<int64_t> expected_shape{1, 1, 224, 224, 16};
  EXPECT_EQ(GeFormatUtil::TransShape(src_desc, FORMAT_NC1HWC0, dst_shape), SUCCESS);
  EXPECT_EQ(dst_shape, expected_shape);
}
}  // namespace ge