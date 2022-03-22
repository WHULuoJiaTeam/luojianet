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

#include <string.h>
#include "graph/utils/file_utils.h"
#include <gtest/gtest.h>
#include "graph/debug/ge_util.h"

namespace ge {
class UtestFileUtils : public testing::Test { 
 public:
  std::string str1 = "/longpath/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  std::string str2 = "/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  std::string str3 = str1 + str2 + str2 + str2 + str2 + str2 + str2;
  const char *str4 = str3.c_str();
  const char *str5 = "/abc/efg";

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestFileUtils, RealPathIsNull) {
  char *path;
  std::string res;
  res = ge::RealPath(path);
  EXPECT_EQ(res, "");
}

TEST_F(UtestFileUtils, RealPathIsNotExist) {
  char *path = "D:/UTTest/aabbccddaaddbcasdaj.txt";
  std::string res;
  res = ge::RealPath(path);
  EXPECT_EQ(res, "");
}

TEST_F(UtestFileUtils, CreateDirectoryPathIsNull) {
  std::string directory_path;
  int32_t ret = ge::CreateDirectory(directory_path);
  EXPECT_EQ(ret, -1);
}

TEST_F(UtestFileUtils, CreateDirectorySuccess) {
  std::string directory_path = "D:\\123\\456";
  int32_t ret = ge::CreateDirectory(directory_path);
  EXPECT_EQ(ret, 0);
}

TEST_F(UtestFileUtils, CreateDirectoryFail) {
  std::string directory_path = "D:/123/456";
  int32_t ret = ge::CreateDirectory(directory_path);
  EXPECT_EQ(ret, -1);
}

TEST_F(UtestFileUtils, CreateDirectoryPathIsGreaterThanMaxPath) {
  std::string directory_path;
  for (int i = 0; i < 4000; i++)
  {
    directory_path.append(std::to_string(i));
  }
  int ret = 0;
  ret = ge::CreateDirectory(directory_path);
  EXPECT_EQ(ret, -1);
}

TEST_F(UtestFileUtils, RealPath) {
  ASSERT_EQ(ge::RealPath(nullptr), "");
  ASSERT_EQ(ge::RealPath(str4), "");
  ASSERT_EQ(ge::RealPath(str5), "");
}

TEST_F(UtestFileUtils, CreateDirectory) {
  ASSERT_EQ(ge::CreateDirectory("~/test"), -1); 
  ASSERT_EQ(ge::CreateDirectory(UtestFileUtils::str3), -1);
}
} // namespace ge
