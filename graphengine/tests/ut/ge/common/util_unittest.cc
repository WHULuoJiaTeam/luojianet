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

#include "common/util.h"

namespace ge {
namespace formats {
class UtestUtilTransfer : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};


INT32 mmAccess2(const CHAR *pathName, INT32 mode)
{
	return -1;
}

TEST_F(UtestUtilTransfer, CheckOutputPathValid) {
  EXPECT_EQ(CheckOutputPathValid("", ""), false);
  EXPECT_EQ(CheckOutputPathValid("", "model"), false);

  char max_file_path[14097] = {0};
  memset(max_file_path, 1, 14097);
  EXPECT_EQ(CheckOutputPathValid(max_file_path, "model"), false);

  EXPECT_EQ(CheckOutputPathValid("$#%", ""), false);

  // system("touch test_util");
  // system("chmod 555 test_util");
  // EXPECT_EQ(CheckOutputPathValid("./test_util", ""), false);
  // system("rm -r test_util");
}

TEST_F(UtestUtilTransfer, CheckInputPathValid) {
  EXPECT_EQ(CheckInputPathValid("", ""), false);
  EXPECT_EQ(CheckInputPathValid("", "model"), false);

  EXPECT_EQ(CheckInputPathValid("$#%", ""), false);

  EXPECT_EQ(CheckInputPathValid("./test_util", ""), false);

}

}
}

