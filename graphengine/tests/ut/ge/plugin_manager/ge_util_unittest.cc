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

#include "common/ge/ge_util.h"

using namespace ge;
using namespace std;

class UtestGeUtil : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

class Foo {
 public:
  int i = 0;

  Foo(int x) { i = x; }

  GE_DELETE_ASSIGN_AND_COPY(Foo);
};

TEST_F(UtestGeUtil, delete_assign_and_copy) {
  Foo f(1);
  ASSERT_EQ(f.i, 1);
}

TEST_F(UtestGeUtil, make_shared) {
  auto f = MakeShared<Foo>(1);
  ASSERT_EQ(f->i, 1);
}
