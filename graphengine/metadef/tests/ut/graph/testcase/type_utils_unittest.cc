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

#include "graph/utils/type_utils.h"
#include <gtest/gtest.h>
#include "graph/debug/ge_util.h"

namespace ge {
class UtestTypeUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestTypeUtils, IsDataTypeValid) {
  ASSERT_FALSE(TypeUtils::IsDataTypeValid(DT_MAX));
  ASSERT_TRUE(TypeUtils::IsDataTypeValid(DT_INT4));

  ASSERT_FALSE(TypeUtils::IsDataTypeValid("MAX"));
  ASSERT_TRUE(TypeUtils::IsDataTypeValid("UINT64"));
  ASSERT_TRUE(TypeUtils::IsDataTypeValid("STRING_REF"));
}

TEST_F(UtestTypeUtils, IsFormatValid) {
  ASSERT_TRUE(TypeUtils::IsFormatValid(FORMAT_NCHW));
  ASSERT_FALSE(TypeUtils::IsFormatValid(FORMAT_END));

  ASSERT_TRUE(TypeUtils::IsFormatValid("DECONV_SP_STRIDE8_TRANS"));
  ASSERT_FALSE(TypeUtils::IsFormatValid("FORMAT_END"));
}

TEST_F(UtestTypeUtils, IsInternalFormat) {
  ASSERT_TRUE(TypeUtils::IsInternalFormat(FORMAT_FRACTAL_Z));
  ASSERT_FALSE(TypeUtils::IsInternalFormat(FORMAT_RESERVED));
}

TEST_F(UtestTypeUtils, FormatToSerialString) {
  ASSERT_EQ(TypeUtils::FormatToSerialString(FORMAT_NCHW), "NCHW");
  ASSERT_EQ(TypeUtils::FormatToSerialString(FORMAT_END), "END");
  ASSERT_EQ(TypeUtils::FormatToSerialString(static_cast<Format>(GetFormatFromSub(FORMAT_FRACTAL_Z, 1))), "FRACTAL_Z:1");
  ASSERT_EQ(TypeUtils::FormatToSerialString(static_cast<Format>(GetFormatFromSub(FORMAT_END, 1))), "END:1");
}

TEST_F(UtestTypeUtils, SerialStringToFormat) {
  ASSERT_EQ(TypeUtils::SerialStringToFormat("NCHW"), FORMAT_NCHW);
  ASSERT_EQ(TypeUtils::SerialStringToFormat("INVALID"), FORMAT_RESERVED);
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:1"), GetFormatFromSub(FORMAT_FRACTAL_Z, 1));
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:01"), GetFormatFromSub(FORMAT_FRACTAL_Z, 1));
  ASSERT_EQ(TypeUtils::SerialStringToFormat("INVALID:1"), FORMAT_RESERVED);
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:1%"), FORMAT_RESERVED);
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:"), FORMAT_RESERVED);  // invalid_argument exception
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:65535"), GetFormatFromSub(FORMAT_FRACTAL_Z, 0xffff));
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:65536"), FORMAT_RESERVED);
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:9223372036854775807"), FORMAT_RESERVED);
}

TEST_F(UtestTypeUtils, DataFormatToFormat) {
  ASSERT_EQ(TypeUtils::DataFormatToFormat("NCHW"), FORMAT_NCHW);
  ASSERT_EQ(TypeUtils::DataFormatToFormat("INVALID"), FORMAT_RESERVED);
  ASSERT_EQ(TypeUtils::DataFormatToFormat("NCHW:1"), GetFormatFromSub(FORMAT_NCHW, 1));
  ASSERT_EQ(TypeUtils::DataFormatToFormat("NCHW:01"), GetFormatFromSub(FORMAT_NCHW, 1));
  ASSERT_EQ(TypeUtils::DataFormatToFormat("INVALID:1"), FORMAT_RESERVED);
  ASSERT_EQ(TypeUtils::DataFormatToFormat("NCHW:1%"), FORMAT_RESERVED);
}

TEST_F(UtestTypeUtils, ImplyTypeToSSerialString) {
  ASSERT_EQ(TypeUtils::ImplyTypeToSerialString(domi::ImplyType::BUILDIN), "buildin");
}

TEST_F(UtestTypeUtils, DataTypeToSerialString) {
  ASSERT_EQ(TypeUtils::DataTypeToSerialString(DT_INT2), "DT_INT2");
  ASSERT_EQ(TypeUtils::DataTypeToSerialString(DT_UINT2), "DT_UINT2");
  ASSERT_EQ(TypeUtils::DataTypeToSerialString(DT_UINT1), "DT_UINT1");
  ASSERT_EQ(TypeUtils::DataTypeToSerialString(DT_MAX), "UNDEFINED");
}

TEST_F(UtestTypeUtils, SerialStringToDataType) {
  ASSERT_EQ(TypeUtils::SerialStringToDataType("DT_UINT1"), DT_UINT1);
  ASSERT_EQ(TypeUtils::SerialStringToDataType("DT_INT2"), DT_INT2);
  ASSERT_EQ(TypeUtils::SerialStringToDataType("DT_MAX"), DT_UNDEFINED);
}

TEST_F(UtestTypeUtils, DomiFormatToFormat) {
  ASSERT_EQ(TypeUtils::DomiFormatToFormat(domi::domiTensorFormat_t::DOMI_TENSOR_NDHWC), FORMAT_NDHWC);
}

TEST_F(UtestTypeUtils, FmkTypeToSerialString) {
  ASSERT_EQ(TypeUtils::FmkTypeToSerialString(domi::FrameworkType::CAFFE), "caffe");
}

TEST_F(UtestTypeUtils, CheckUint64MulOverflow) {
  ASSERT_FALSE(TypeUtils::CheckUint64MulOverflow(0x00ULL, 0x00UL));
  ASSERT_FALSE(TypeUtils::CheckUint64MulOverflow(0x02ULL, 0x01UL));
  ASSERT_TRUE(TypeUtils::CheckUint64MulOverflow(0xFFFFFFFFFFFFULL, 0xFFFFFFFUL));
}

TEST_F(UtestTypeUtils, ImplyTypeToSerialString) {
  ASSERT_EQ(TypeUtils::ImplyTypeToSerialString(domi::ImplyType::BUILDIN), "buildin");
}

TEST_F(UtestTypeUtils, DomiFormatToFormat2) {
  ASSERT_EQ(TypeUtils::DomiFormatToFormat(domi::DOMI_TENSOR_NCHW), FORMAT_NCHW);
  ASSERT_EQ(TypeUtils::DomiFormatToFormat(domi::DOMI_TENSOR_RESERVED), FORMAT_RESERVED);
}

TEST_F(UtestTypeUtils, CheckUint64MulOverflow2) {
  ASSERT_FALSE(TypeUtils::CheckUint64MulOverflow(0, 1));
  ASSERT_FALSE(TypeUtils::CheckUint64MulOverflow(1, 1));
  ASSERT_TRUE(TypeUtils::CheckUint64MulOverflow(ULLONG_MAX, 2));
}

TEST_F(UtestTypeUtils, FmkTypeToSerialString2) {
  ASSERT_EQ(TypeUtils::FmkTypeToSerialString(domi::CAFFE), "caffe");
  ASSERT_EQ(TypeUtils::FmkTypeToSerialString(static_cast<domi::FrameworkType>(domi::FRAMEWORK_RESERVED + 1)), "");
}
}
