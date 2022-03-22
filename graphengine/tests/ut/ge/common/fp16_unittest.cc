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

#include "common/fp16_t.h"

namespace ge {
namespace formats {
class UtestFP16 : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestFP16, fp16_to_other) {
  fp16_t test;
  float num = test.ToFloat();
  EXPECT_EQ(num, 0.0);

  double num2 = test.ToDouble();
  EXPECT_EQ(num2, 0);

  int16_t num3 = test.ToInt16();
  EXPECT_EQ(num3, 0);

  int32_t num4 = test.ToInt32();
  EXPECT_EQ(num4, 0);

  int8_t num5 = test.ToInt8();
  EXPECT_EQ(num5, 0);

  uint16_t num6 = test.ToUInt16();
  EXPECT_EQ(num6, 0);

  uint32_t num7 = test.ToUInt16();
  EXPECT_EQ(num7, 0);

  uint8_t num8 = test.ToUInt8();
  EXPECT_EQ(num8, 0);
}
}  // namespace formats
}  // namespace ge
