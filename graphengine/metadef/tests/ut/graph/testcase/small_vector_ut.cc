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
#include "graph/small_vector.h"
#include "test_structs.h"
#include "func_counter.h"
namespace ge {
class SmallVectorUt : public testing::Test {};

TEST_F(SmallVectorUt, ConstructAndFree) {
  std::vector<InlineStructB> v1;
  std::vector<InlineStructB> v2(std::move(v1));
  SmallVector<InlineStructB, 5> vec1;
  SmallVector<InlineStructB, 5> vec2(2);
  SmallVector<InlineStructB, 5> vec3(5);
  SmallVector<InlineStructB, 5> vec4(6);
  SmallVector<InlineStructB, 5> vec5(10);
  SmallVector<InlineStructB, 5> vec6({InlineStructB(), InlineStructB()});
}

TEST_F(SmallVectorUt, Construct_CallCorrectConstructor1) {
  FuncCounter::Clear();
  SmallVector<FuncCounter, 5> vec1;
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  FuncCounter fc;
  FuncCounter::Clear();
  SmallVector<FuncCounter, 5> vec2(10, fc);
  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 10);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  FuncCounter::Clear();
  SmallVector<FuncCounter, 5> vec3(vec2);
  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 10);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  FuncCounter::Clear();
  SmallVector<FuncCounter, 5> vec4(std::move(vec2));  // 直接挪指针
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(SmallVectorUt, Construct_CallCorrectConstructor2) {
  FuncCounter fc;
  SmallVector<FuncCounter, 5> vec2(3, fc);

  FuncCounter::Clear();
  SmallVector<FuncCounter, 5> vec3(std::move(vec2));
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 3);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 3);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(SmallVectorUt, CopyConstructorAndFree) {
  SmallVector<InlineStructB, 5> vec1;
  SmallVector<InlineStructB, 5> vec2(2);
  SmallVector<InlineStructB, 5> vec3(5);
  SmallVector<InlineStructB, 5> vec4(6);
  SmallVector<InlineStructB, 5> vec5(10);

  SmallVector<InlineStructB, 5> vec2_1(vec1);
  SmallVector<InlineStructB, 5> vec2_2(vec2);
  SmallVector<InlineStructB, 5> vec2_3(vec3);
  SmallVector<InlineStructB, 5> vec2_4(vec4);
  SmallVector<InlineStructB, 5> vec2_5(vec5);
}

TEST_F(SmallVectorUt, ConstructorCorrectSizeAndCap) {
  std::vector<int32_t> veccccc;
  SmallVector<int32_t, 5> vec1;
  EXPECT_EQ(vec1.size(), 0);
  EXPECT_EQ(vec1.capacity(), 5);

  SmallVector<int32_t, 5> vec2(2);
  EXPECT_EQ(vec2.size(), 2);
  EXPECT_EQ(vec2.capacity(), 5);

  SmallVector<int32_t, 5> vec2_10(2, 10);
  EXPECT_EQ(vec2.size(), 2);
  EXPECT_EQ(vec2.capacity(), 5);
  int32_t expect_vec2_10[2] = {10, 10};
  EXPECT_EQ(memcmp(vec2_10.data(), expect_vec2_10, sizeof(expect_vec2_10)), 0);

  SmallVector<int32_t, 5> vec4(6, 10);
  EXPECT_EQ(vec4.size(), 6);
  EXPECT_GE(vec4.capacity(), 6);
  int32_t expect_vec4_10[6] = {10, 10, 10, 10, 10, 10};
  EXPECT_EQ(memcmp(vec4.data(), expect_vec4_10, sizeof(expect_vec4_10)), 0);

  SmallVector<int32_t, 5> vec6({1,2,3});
  EXPECT_EQ(vec6.size(), 3);
  EXPECT_EQ(vec6.capacity(), 5);
  int32_t expect_vec6_10[] = {1,2,3};
  EXPECT_EQ(memcmp(vec6.data(), expect_vec6_10, sizeof(expect_vec6_10)), 0);

  SmallVector<int32_t, 5> vec7({1,2,3,4,5,6});
  EXPECT_EQ(vec7.size(), 6);
  EXPECT_GE(vec7.capacity(), 6);
  int32_t expect_vec7_10[] = {1,2,3,4,5,6};
  EXPECT_EQ(memcmp(vec7.data(), expect_vec7_10, sizeof(expect_vec7_10)), 0);
}

TEST_F(SmallVectorUt, MoveConstructor) {
  SmallVector<InlineStructB, 5> vec1;
  SmallVector<InlineStructB, 5> vec2(2);
  SmallVector<InlineStructB, 5> vec3(5);
  SmallVector<InlineStructB, 5> vec4(6);
  SmallVector<InlineStructB, 5> vec5(10);

  SmallVector<InlineStructB, 5> vec2_1(std::move(vec1));
  SmallVector<InlineStructB, 5> vec2_2(std::move(vec2));
  SmallVector<InlineStructB, 5> vec2_3(std::move(vec3));
  SmallVector<InlineStructB, 5> vec2_4(std::move(vec4));
  SmallVector<InlineStructB, 5> vec2_5(std::move(vec5));
}

TEST_F(SmallVectorUt, MoveAssign) {
  SmallVector<InlineStructB, 5> vec1;
  SmallVector<InlineStructB, 5> vec2(2);
  SmallVector<InlineStructB, 5> vec3(5);
  SmallVector<InlineStructB, 5> vec4(6);
  SmallVector<InlineStructB, 5> vec5(10);

  SmallVector<InlineStructB, 5> vec2_1;
  SmallVector<InlineStructB, 5> vec2_6;

  vec2_1 = std::move(vec1);
  vec2_1 = std::move(vec3);
  vec2_1 = std::move(vec2);

  vec2_6 = SmallVector<InlineStructB, 5>(20);
  vec2_6 = std::move(vec4);
  vec2_6 = std::move(vec5);
  vec2_6 = SmallVector<InlineStructB, 5>(3);
}

TEST_F(SmallVectorUt, CopyAssignInlineCap) {
  SmallVector<InlineStructB, 5> vec1;
  SmallVector<InlineStructB, 5> vec2(2);
  SmallVector<InlineStructB, 5> vec3(5);
  SmallVector<InlineStructB, 5> vec4(6);
  SmallVector<InlineStructB, 5> vec5(10);

  SmallVector<InlineStructB, 5> vec6;

  // 只要dst size不大于5，不论size变大还是变小，那么vec6的cap不会变化
  vec6 = vec1;
  EXPECT_EQ(vec6.capacity(), 5);
  EXPECT_EQ(vec6.size(), 0);
  vec6 = vec3;
  EXPECT_EQ(vec6.capacity(), 5);
  EXPECT_EQ(vec6.size(), 5);
  vec6 = vec2;
  EXPECT_EQ(vec6.capacity(), 5);
  EXPECT_EQ(vec6.size(), 2);
}

TEST_F(SmallVectorUt, CopyAssignInlineToAlloc) {
  SmallVector<InlineStructB, 5> vec1;
  SmallVector<InlineStructB, 5> vec2(2);
  SmallVector<InlineStructB, 5> vec3(5);
  SmallVector<InlineStructB, 5> vec4(6);
  SmallVector<InlineStructB, 5> vec5(10);

  SmallVector<InlineStructB, 5> vec6(3);
  vec6 = vec5;
  EXPECT_EQ(vec6.capacity(), 10);
  EXPECT_EQ(vec6.size(), 10);
}

TEST_F(SmallVectorUt, CopyAssignAllocToInline) {
  SmallVector<InlineStructB, 5> vec1;
  SmallVector<InlineStructB, 5> vec2(2);
  SmallVector<InlineStructB, 5> vec3(5);
  SmallVector<InlineStructB, 5> vec4(6);
  SmallVector<InlineStructB, 5> vec5(10);

  SmallVector<InlineStructB, 5> vec6(10);

  // 为了减少内存申请和释放的次数，即使vec6使用allocated_storage，size降回到N以下时，也不会用回inline_storage了
  vec6 = vec2;
  EXPECT_EQ(vec6.capacity(), 10);
  EXPECT_EQ(vec6.size(), 2);

  vec6 = vec4;
  EXPECT_EQ(vec6.capacity(), 10);
  EXPECT_EQ(vec6.size(), 6);
}

TEST_F(SmallVectorUt, CopyAssignAllocExpand) {
  SmallVector<InlineStructB, 5> vec3(8);
  SmallVector<InlineStructB, 5> vec4(10);
  SmallVector<InlineStructB, 5> vec5(20);

  SmallVector<InlineStructB, 5> vec6(9);

  vec6 = vec3;
  EXPECT_EQ(vec6.capacity(), 9);
  EXPECT_EQ(vec6.size(), 8);

  vec6 = vec5;
  EXPECT_EQ(vec6.capacity(), 20);
  EXPECT_EQ(vec6.size(), 20);

  vec6 = vec4;
  EXPECT_EQ(vec6.capacity(), 20);
  EXPECT_EQ(vec6.size(), 10);
}

TEST_F(SmallVectorUt, CopyAssignOk1) {
  SmallVector<std::vector<int64_t>, 100> sv1;
  sv1.emplace_back(10, 100);
  sv1.emplace_back(10, 200);

  SmallVector<std::vector<int64_t>, 100> sv2 = sv1;
  sv1[0].push_back(100);
  EXPECT_EQ(sv2.size(), sv1.size());
  EXPECT_NE(sv2[0], sv1[0]);
  EXPECT_EQ(sv2[1], sv1[1]);
}

TEST_F(SmallVectorUt, Assign_CallCorrectConstructor) {
  SmallVector<FuncCounter, 5> vec1(5);
  SmallVector<FuncCounter, 5> vec2;
  SmallVector<FuncCounter, 5> vec3;

  FuncCounter::Clear();
  vec2 = vec1;
  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 5);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  FuncCounter::Clear();
  vec3 = std::move(vec1);
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 5);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 5);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  FuncCounter::Clear();
  vec2 = vec3;
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 5);
  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 5);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  FuncCounter::Clear();
  vec2 = std::move(vec3);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 10);
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 5);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(SmallVectorUt, Clear_CallCorrectConstructor) {
  SmallVector<FuncCounter, 5> vec1(5);
  SmallVector<FuncCounter, 5> vec2;

  FuncCounter::Clear();
  vec1.clear();
  EXPECT_EQ(FuncCounter::destruct_times, 5);
  FuncCounter::Clear();
  vec2.clear();
  EXPECT_EQ(FuncCounter::destruct_times, 0);
}

TEST_F(SmallVectorUt, Insert_CallCorrectConstructor) {
  SmallVector<FuncCounter, 10> vec1(5);

  FuncCounter fc;
  FuncCounter::Clear();
  vec1.insert(vec1.end(), fc);
  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 1);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  FuncCounter::Clear();
  vec1.insert(vec1.begin(), fc);
  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 1);
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 6);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 6);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(SmallVectorUt, InsertExpand_CallCorrectConstructor) {
  SmallVector<FuncCounter, 5> vec1(5);

  FuncCounter fc;
  FuncCounter::Clear();
  vec1.insert(vec1.end(), fc);
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 5);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 5);
  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 1);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(SmallVectorUt, MoveInsert_CallCorrectConstructor) {
  SmallVector<FuncCounter, 10> vec1(5);

  FuncCounter fc1;
  FuncCounter::Clear();
  vec1.insert(vec1.end(), std::move(fc1));
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 1);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  FuncCounter fc2;
  FuncCounter::Clear();
  vec1.insert(vec1.begin(), std::move(fc2));
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 7);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 6);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(SmallVectorUt, InsertMultiple_CallCorrectConstructor) {
  SmallVector<FuncCounter, 20> vec1(5);

  FuncCounter fc;
  FuncCounter::Clear();
  vec1.insert(vec1.end(), 3, fc);
  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 3);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  FuncCounter::Clear();
  vec1.insert(vec1.begin(), 3, fc);
  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 3);
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 8);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 8);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}
TEST_F(SmallVectorUt, ClearOk) {
  SmallVector<InlineStructB, 5> vec1;
  SmallVector<InlineStructB, 5> vec2(2);
  SmallVector<InlineStructB, 5> vec3(5);
  SmallVector<InlineStructB, 5> vec4(6);
  SmallVector<InlineStructB, 5> vec5(10);

  vec1.clear();
  EXPECT_EQ(vec1.size(), 0);
  EXPECT_EQ(vec1.capacity(), 5);

  vec2.clear();
  EXPECT_EQ(vec2.size(), 0);
  EXPECT_EQ(vec2.capacity(), 5);

  vec3.clear();
  EXPECT_EQ(vec3.size(), 0);
  EXPECT_EQ(vec3.capacity(), 5);

  vec4.clear();
  EXPECT_EQ(vec4.size(), 0);
  EXPECT_EQ(vec4.capacity(), 5);

  vec5.clear();
  EXPECT_EQ(vec5.size(), 0);
  EXPECT_EQ(vec5.capacity(), 5);
}

TEST_F(SmallVectorUt, At) {
  SmallVector<InlineStructB, 2> vec2(2);
  SmallVector<InlineStructB, 2> vec3(3);

  for (int32_t i = 0; i < 10; ++i) {
    vec2.at(0).Set(i, i);
    vec2.at(1).Set(i, i * 10 + 1);

    vec3.at(0).Set(i, i * 100 + 10);
    vec3.at(1).Set(i, i * 100 + 11);
    vec3.at(2).Set(i, i * 100 + 12);
  }

  const SmallVector<InlineStructB, 2> &read_vec2 = vec2;
  const SmallVector<InlineStructB, 2> &read_vec3 = vec3;

  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(read_vec2.at(0).Get(i), i);
    EXPECT_EQ(read_vec2.at(1).Get(i), i * 10 + 1);

    EXPECT_EQ(read_vec3.at(0).Get(i), i * 100 + 10);
    EXPECT_EQ(read_vec3.at(1).Get(i), i * 100 + 11);
    EXPECT_EQ(read_vec3.at(2).Get(i), i * 100 + 12);
  }
}

TEST_F(SmallVectorUt, BeginAndEnd) {
  SmallVector<InlineStructB, 2> vec0;
  SmallVector<InlineStructB, 2> vec1(1);
  SmallVector<InlineStructB, 2> vec2(2);
  SmallVector<InlineStructB, 2> vec3(3);

  size_t iter_count = 0;
  for (auto iter = vec3.begin(); iter != vec3.end(); ++iter) {
    EXPECT_EQ(&*iter, &vec3.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 3);

  iter_count = 0;
  for (auto iter = vec2.begin(); iter != vec2.end(); ++iter) {
    EXPECT_EQ(&*iter, &vec2.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 2);

  iter_count = 0;
  for (auto iter = vec1.begin(); iter != vec1.end(); ++iter) {
    EXPECT_EQ(&*iter, &vec1.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 1);

  iter_count = 0;
  for (auto iter = vec0.begin(); iter != vec0.end(); ++iter) {
    iter_count++;
  }
  EXPECT_EQ(iter_count, 0);
}

TEST_F(SmallVectorUt, CBeginAndCEnd) {
  SmallVector<InlineStructB, 2> vec0;
  SmallVector<InlineStructB, 2> vec1(1);
  SmallVector<InlineStructB, 2> vec2(2);
  SmallVector<InlineStructB, 2> vec3(3);

  size_t iter_count = 0;
  for (auto iter = vec3.cbegin(); iter != vec3.cend(); ++iter) {
    EXPECT_EQ(&*iter, &vec3.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 3);

  iter_count = 0;
  for (auto iter = vec2.cbegin(); iter != vec2.cend(); ++iter) {
    EXPECT_EQ(&*iter, &vec2.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 2);

  iter_count = 0;
  for (auto iter = vec1.cbegin(); iter != vec1.cend(); ++iter) {
    EXPECT_EQ(&*iter, &vec1.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 1);

  iter_count = 0;
  for (auto iter = vec0.cbegin(); iter != vec0.cend(); ++iter) {
    iter_count++;
  }
  EXPECT_EQ(iter_count, 0);
}

TEST_F(SmallVectorUt, BeginAndEnd_Const) {
  const SmallVector<InlineStructB, 2> vec0;
  const SmallVector<InlineStructB, 2> vec1(1);
  const SmallVector<InlineStructB, 2> vec2(2);
  const SmallVector<InlineStructB, 2> vec3(3);

  size_t iter_count = 0;
  for (auto iter = vec3.begin(); iter != vec3.end(); ++iter) {
    EXPECT_EQ(&*iter, &vec3.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 3);

  iter_count = 0;
  for (auto iter = vec2.begin(); iter != vec2.end(); ++iter) {
    EXPECT_EQ(&*iter, &vec2.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 2);

  iter_count = 0;
  for (auto iter = vec1.begin(); iter != vec1.end(); ++iter) {
    EXPECT_EQ(&*iter, &vec1.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 1);

  iter_count = 0;
  for (auto iter = vec0.begin(); iter != vec0.end(); ++iter) {
    iter_count++;
  }
  EXPECT_EQ(iter_count, 0);
}

TEST_F(SmallVectorUt, CBeginAndCEnd_Const) {
  const SmallVector<InlineStructB, 2> vec0;
  const SmallVector<InlineStructB, 2> vec1(1);
  const SmallVector<InlineStructB, 2> vec2(2);
  const SmallVector<InlineStructB, 2> vec3(3);

  size_t iter_count = 0;
  for (auto iter = vec3.cbegin(); iter != vec3.cend(); ++iter) {
    EXPECT_EQ(&*iter, &vec3.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 3);

  iter_count = 0;
  for (auto iter = vec2.cbegin(); iter != vec2.cend(); ++iter) {
    EXPECT_EQ(&*iter, &vec2.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 2);

  iter_count = 0;
  for (auto iter = vec1.cbegin(); iter != vec1.cend(); ++iter) {
    EXPECT_EQ(&*iter, &vec1.at(iter_count));
    iter_count++;
  }
  EXPECT_EQ(iter_count, 1);

  iter_count = 0;
  for (auto iter = vec0.cbegin(); iter != vec0.cend(); ++iter) {
    iter_count++;
  }
  EXPECT_EQ(iter_count, 0);
}

TEST_F(SmallVectorUt, Front1) {
  const SmallVector<InlineStructB, 4> vec1(1);
  const SmallVector<InlineStructB, 4> vec2(4);
  const SmallVector<InlineStructB, 4> vec3(8);
  EXPECT_EQ(&vec1.front(), &vec1.at(0));
  EXPECT_EQ(&vec2.front(), &vec2.at(0));
  EXPECT_EQ(&vec3.front(), &vec3.at(0));

  SmallVector<InlineStructB, 4> vec1_1(1);
  SmallVector<InlineStructB, 4> vec1_2(4);
  SmallVector<InlineStructB, 4> vec1_3(8);
  EXPECT_EQ(&vec1_1.front(), &vec1_1.at(0));
  EXPECT_EQ(&vec1_2.front(), &vec1_2.at(0));
  EXPECT_EQ(&vec1_3.front(), &vec1_3.at(0));
}

TEST_F(SmallVectorUt, Back1) {
  const SmallVector<InlineStructB, 4> vec1(1);
  const SmallVector<InlineStructB, 4> vec2(4);
  const SmallVector<InlineStructB, 4> vec3(8);
  EXPECT_EQ(&vec1.back(), &vec1.at(0));
  EXPECT_EQ(&vec2.back(), &vec2.at(3));
  EXPECT_EQ(&vec3.back(), &vec3.at(7));

  SmallVector<InlineStructB, 4> vec1_1(1);
  SmallVector<InlineStructB, 4> vec1_2(4);
  SmallVector<InlineStructB, 4> vec1_3(8);
  EXPECT_EQ(&vec1_1.back(), &vec1_1.at(0));
  EXPECT_EQ(&vec1_2.back(), &vec1_2.at(3));
  EXPECT_EQ(&vec1_3.back(), &vec1_3.at(7));
}

TEST_F(SmallVectorUt, FrontAndBack) {
  const SmallVector<InlineStructB, 4> vec1(1);
  const SmallVector<InlineStructB, 4> vec2(4);
  const SmallVector<InlineStructB, 4> vec3(8);
  EXPECT_EQ(&vec1.front(), &vec1.back());
  EXPECT_NE(&vec2.front(), &vec2.back());
  EXPECT_NE(&vec3.front(), &vec3.back());

  SmallVector<InlineStructB, 4> vec1_1(1);
  SmallVector<InlineStructB, 4> vec1_2(4);
  SmallVector<InlineStructB, 4> vec1_3(8);
  EXPECT_EQ(&vec1_1.front(), &vec1_1.back());
  EXPECT_NE(&vec1_2.front(), &vec1_2.back());
  EXPECT_NE(&vec1_3.front(), &vec1_3.back());
}

TEST_F(SmallVectorUt, RCIter_Const) {
  const SmallVector<InlineStructB, 2> vec0;
  const SmallVector<InlineStructB, 2> vec1(1);
  const SmallVector<InlineStructB, 2> vec2(2);
  const SmallVector<InlineStructB, 2> vec3(3);

  size_t iter_count = 3;
  for (auto iter = vec3.crbegin(); iter != vec3.crend(); ++iter) {
    EXPECT_EQ(&*iter, &vec3.at(--iter_count));
  }
  EXPECT_EQ(iter_count, 0);

  iter_count = 2;
  for (auto iter = vec2.crbegin(); iter != vec2.crend(); ++iter) {
    EXPECT_EQ(&*iter, &vec2.at(--iter_count));
  }
  EXPECT_EQ(iter_count, 0);

  iter_count = 1;
  for (auto iter = vec1.crbegin(); iter != vec1.crend(); ++iter) {
    EXPECT_EQ(&*iter, &vec1.at(--iter_count));
  }
  EXPECT_EQ(iter_count, 0);

  iter_count = 0;
  for (auto iter = vec0.crbegin(); iter != vec0.crend(); ++iter) {
    iter_count++;
  }
  EXPECT_EQ(iter_count, 0);
}

TEST_F(SmallVectorUt, RCIter_Const1) {
  const SmallVector<InlineStructB, 2> vec0;
  const SmallVector<InlineStructB, 2> vec1(1);
  const SmallVector<InlineStructB, 2> vec2(2);
  const SmallVector<InlineStructB, 2> vec3(3);

  size_t iter_count = 3;
  for (auto iter = vec3.rbegin(); iter != vec3.rend(); ++iter) {
    EXPECT_EQ(&*iter, &vec3.at(--iter_count));
  }
  EXPECT_EQ(iter_count, 0);

  iter_count = 2;
  for (auto iter = vec2.rbegin(); iter != vec2.rend(); ++iter) {
    EXPECT_EQ(&*iter, &vec2.at(--iter_count));
  }
  EXPECT_EQ(iter_count, 0);

  iter_count = 1;
  for (auto iter = vec1.rbegin(); iter != vec1.rend(); ++iter) {
    EXPECT_EQ(&*iter, &vec1.at(--iter_count));
  }
  EXPECT_EQ(iter_count, 0);

  iter_count = 0;
  for (auto iter = vec0.rbegin(); iter != vec0.rend(); ++iter) {
    iter_count++;
  }
  EXPECT_EQ(iter_count, 0);
}

TEST_F(SmallVectorUt, RCIter1) {
  SmallVector<InlineStructB, 2> vec0;
  SmallVector<InlineStructB, 2> vec1(1);
  SmallVector<InlineStructB, 2> vec2(2);
  SmallVector<InlineStructB, 2> vec3(3);

  size_t iter_count = 3;
  for (auto iter = vec3.rbegin(); iter != vec3.rend(); ++iter) {
    EXPECT_EQ(&*iter, &vec3.at(--iter_count));
  }
  EXPECT_EQ(iter_count, 0);

  iter_count = 2;
  for (auto iter = vec2.rbegin(); iter != vec2.rend(); ++iter) {
    EXPECT_EQ(&*iter, &vec2.at(--iter_count));
  }
  EXPECT_EQ(iter_count, 0);

  iter_count = 1;
  for (auto iter = vec1.rbegin(); iter != vec1.rend(); ++iter) {
    EXPECT_EQ(&*iter, &vec1.at(--iter_count));
  }
  EXPECT_EQ(iter_count, 0);

  iter_count = 0;
  for (auto iter = vec0.rbegin(); iter != vec0.rend(); ++iter) {
    iter_count++;
  }
  EXPECT_EQ(iter_count, 0);
}

TEST_F(SmallVectorUt, EmptyOk) {
  const SmallVector<InlineStructB, 2> vec0;
  const SmallVector<InlineStructB, 2> vec1(1);
  const SmallVector<InlineStructB, 2> vec2(2);
  const SmallVector<InlineStructB, 2> vec3(3);

  EXPECT_TRUE(vec0.empty());
  EXPECT_FALSE(vec1.empty());
  EXPECT_FALSE(vec2.empty());
  EXPECT_FALSE(vec3.empty());
}

TEST_F(SmallVectorUt, SizeOk) {
  const SmallVector<InlineStructB, 2> vec0;
  const SmallVector<InlineStructB, 2> vec1(1);
  const SmallVector<InlineStructB, 2> vec2(2);
  const SmallVector<InlineStructB, 2> vec3(3);

  EXPECT_EQ(vec0.size(), 0);
  EXPECT_EQ(vec1.size(), 1);
  EXPECT_EQ(vec2.size(), 2);
  EXPECT_EQ(vec3.size(), 3);
}

void RandomB(InlineStructB &b) {
  for (int32_t i = 0; i < 10; ++i) {
    b.Set(i, rand());
  }
}

TEST_F(SmallVectorUt, InsertFront) {
  SmallVector<InlineStructB, 2> vec0;
  SmallVector<InlineStructB, 2> vec1(1);
  SmallVector<InlineStructB, 2> vec2(2);
  SmallVector<InlineStructB, 2> vec3(3);

  InlineStructB b;
  for (int32_t i = 0; i < 10; ++i) {
    b.Set(i, i * 10);
  }
  InlineStructB b1;
  RandomB(b1);
  InlineStructB b2;
  RandomB(b2);

  auto iter = vec0.insert(vec0.cbegin(), b);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec0.size(), 1);

  vec1[0] = b1;
  iter = vec1.insert(vec1.cbegin(), b);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec1[0], b);
  EXPECT_EQ(vec1[1], b1);
  EXPECT_EQ(vec1.size(), 2);

  vec2[0] = b1;
  vec2[1] = b2;
  iter = vec2.insert(vec2.cbegin(), b);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec2[0], b);
  EXPECT_EQ(vec2[1], b1);
  EXPECT_EQ(vec2[2], b2);
  EXPECT_EQ(vec2.size(), 3);

  vec3[1] = b1;
  vec3[2] = b2;
  iter = vec3.insert(vec3.cbegin(), b);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec3[0], b);
  EXPECT_EQ(vec3[2], b1);
  EXPECT_EQ(vec3[3], b2);
  EXPECT_EQ(vec3.size(), 4);
}

TEST_F(SmallVectorUt, InsertEnd) {
  SmallVector<InlineStructB, 2> vec0;
  SmallVector<InlineStructB, 2> vec1(1);
  SmallVector<InlineStructB, 2> vec2(2);
  SmallVector<InlineStructB, 2> vec3(3);

  InlineStructB b;
  for (int32_t i = 0; i < 10; ++i) {
    b.Set(i, i * 10);
  }
  InlineStructB b1;
  RandomB(b1);
  InlineStructB b2;
  RandomB(b2);

  auto iter = vec0.insert(vec0.end(), b);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec0.size(), 1);

  vec1[0] = b1;
  iter = vec1.insert(vec1.end(), b);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec1[1], b);
  EXPECT_EQ(vec1[0], b1);
  EXPECT_EQ(vec1.size(), 2);

  vec2[0] = b1;
  vec2[1] = b2;
  iter = vec2.insert(vec2.end(), b);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec2[2], b);
  EXPECT_EQ(vec2[0], b1);
  EXPECT_EQ(vec2[1], b2);
  EXPECT_EQ(vec2.size(), 3);

  vec3[1] = b1;
  vec3[2] = b2;
  iter = vec3.insert(vec3.end(), b);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec3[3], b);
  EXPECT_EQ(vec3[1], b1);
  EXPECT_EQ(vec3[2], b2);
  EXPECT_EQ(vec3.size(), 4);
}

TEST_F(SmallVectorUt, InsertMid) {
  SmallVector<InlineStructB, 2> vec2(2);
  SmallVector<InlineStructB, 2> vec3(3);

  InlineStructB b;
  for (int32_t i = 0; i < 10; ++i) {
    b.Set(i, i * 10);
  }
  InlineStructB b1;
  RandomB(b1);
  InlineStructB b2;
  RandomB(b2);

  vec2[0] = b1;
  vec2[1] = b2;
  auto iter = vec2.insert(vec2.begin() + 1, b);  // b1, b, b2
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec2[1], b);
  EXPECT_EQ(vec2[0], b1);
  EXPECT_EQ(vec2[2], b2);
  EXPECT_EQ(vec2.size(), 3);

  vec3[1] = b1;
  vec3[2] = b2;
  iter = vec3.insert(vec3.begin() + 1, b);  // xx, b, b1, b2
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec3[1], b);
  EXPECT_EQ(vec3[2], b1);
  EXPECT_EQ(vec3[3], b2);
  EXPECT_EQ(vec3.size(), 4);
}

TEST_F(SmallVectorUt, InsertMid_Move) {
  SmallVector<InlineStructB, 2> vec3(3);

  InlineStructB b;
  for (int32_t i = 0; i < 10; ++i) {
    b.Set(i, i * 10);
  }
  InlineStructB b1;
  RandomB(b1);
  InlineStructB b2;
  RandomB(b2);
  InlineStructB b_back = b;
  auto b_back_p = b.GetP();

  vec3[1] = b1;
  vec3[2] = b2;
  auto iter = vec3.insert(vec3.begin() + 1, std::move(b));  // xx, b, b1, b2
  EXPECT_EQ(vec3.size(), 4);
  EXPECT_EQ(*iter, b_back);
  EXPECT_EQ(iter->GetP(), b_back_p);
  EXPECT_EQ(vec3[1], b_back);
  EXPECT_EQ(vec3[2], b1);
  EXPECT_EQ(vec3[3], b2);
}

TEST_F(SmallVectorUt, InsertMid_Multiple) {
  SmallVector<std::vector<int64_t>, 3> vec2(2);
  SmallVector<std::vector<int64_t>, 3> vec3(3);
  SmallVector<std::vector<int64_t>, 3> vec4(4);

  std::vector<int64_t> b{1, 2, 3, 4, 5};
  std::vector<int64_t> b1{6, 7, 8, 9, 10};
  std::vector<int64_t> b2{11, 12, 13, 14, 15};

  vec2[0] = b1;
  vec2[1] = b2;
  auto iter = vec2.insert(vec2.begin() + 1, 3, b);  // b1, b, b, b, b2
  EXPECT_EQ(vec2.size(), 5);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec2[0], b1);
  EXPECT_EQ(vec2[1], b);
  EXPECT_EQ(vec2[2], b);
  EXPECT_EQ(vec2[3], b);
  EXPECT_EQ(vec2[4], b2);

  vec3[1] = b1;
  vec3[2] = b2;
  iter = vec3.insert(vec3.begin() + 1, 3, b);  // xx, b, b, b, b1, b2
  EXPECT_EQ(vec3.size(), 6);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec3[1], b);
  EXPECT_EQ(vec3[2], b);
  EXPECT_EQ(vec3[3], b);
  EXPECT_EQ(vec3[4], b1);
  EXPECT_EQ(vec3[5], b2);

  vec4[0] = b1;
  vec4[1] = b2;
  vec4[2] = b1;
  vec4[3] = b2;
  iter = vec4.insert(vec4.begin() + 2, 1, b);  // b1,b2,b,b1,b2
  EXPECT_EQ(vec4.size(), 5);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec4[0], b1);
  EXPECT_EQ(vec4[1], b2);
  EXPECT_EQ(vec4[2], b);
  EXPECT_EQ(vec4[3], b1);
  EXPECT_EQ(vec4[4], b2);
}

TEST_F(SmallVectorUt, InsertMid_Multiple1) {
  SmallVector<InlineStructB, 3> vec2(2);
  SmallVector<InlineStructB, 3> vec3(3);
  SmallVector<InlineStructB, 3> vec4(4);

  InlineStructB b;
  for (int32_t i = 0; i < 10; ++i) {
    b.Set(i, i * 10);
  }
  InlineStructB b1;
  RandomB(b1);
  InlineStructB b2;
  RandomB(b2);

  vec2[0] = b1;
  vec2[1] = b2;
  auto iter = vec2.insert(vec2.begin() + 1, 3, b);  // b1, b, b, b, b2
  EXPECT_EQ(vec2.size(), 5);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec2[0], b1);
  EXPECT_EQ(vec2[1], b);
  EXPECT_EQ(vec2[2], b);
  EXPECT_EQ(vec2[3], b);
  EXPECT_EQ(vec2[4], b2);

  vec3[1] = b1;
  vec3[2] = b2;
  iter = vec3.insert(vec3.begin() + 1, 3, b);  // xx, b, b, b, b1, b2
  EXPECT_EQ(vec3.size(), 6);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec3[1], b);
  EXPECT_EQ(vec3[2], b);
  EXPECT_EQ(vec3[3], b);
  EXPECT_EQ(vec3[4], b1);
  EXPECT_EQ(vec3[5], b2);

  vec4[0] = b1;
  vec4[1] = b2;
  vec4[2] = b1;
  vec4[3] = b2;
  iter = vec4.insert(vec4.begin() + 2, 1, b);  // b1,b2,b,b1,b2
  EXPECT_EQ(vec4.size(), 5);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec4[0], b1);
  EXPECT_EQ(vec4[1], b2);
  EXPECT_EQ(vec4[2], b);
  EXPECT_EQ(vec4[3], b1);
  EXPECT_EQ(vec4[4], b2);
}

TEST_F(SmallVectorUt, InsertMid_Multiple2) {
  SmallVector<std::vector<int64_t>, 3> vec2(2);

  std::vector<int64_t> b{1, 2, 3, 4, 5};
  std::vector<int64_t> b1{6, 7, 8, 9, 10};
  std::vector<int64_t> b2{11, 12, 13, 14, 15};

  vec2[0] = b1;
  vec2[1] = b2;
  auto iter = vec2.insert(vec2.begin() + 1, 8, b);  // b1, b[8], b2
  EXPECT_EQ(vec2.size(), 10);
  EXPECT_GE(vec2.capacity(), 10);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec2[0], b1);
  for (auto i = 0; i < 8; ++i) {
    EXPECT_EQ(vec2[i + 1], b);
  }
  EXPECT_EQ(vec2[9], b2);
}

TEST_F(SmallVectorUt, InsertMid_Multiple_NotExpandCap) {
  SmallVector<std::vector<int64_t>, 8> vec2(4);

  std::vector<int64_t> b{1, 2, 3, 4, 5};
  std::vector<int64_t> b1{6, 7, 8, 9, 10};
  std::vector<int64_t> b2{11, 12, 13, 14, 15};
  std::vector<int64_t> b3{16, 17, 18, 19, 20};
  std::vector<int64_t> b4{21, 22, 13, 14, 15};
  vec2[0] = b1;
  vec2[1] = b2;
  vec2[2] = b3;
  vec2[3] = b4;
  auto iter = vec2.insert(vec2.begin() + 1, 4, b);  // b1, b[4], b2
  EXPECT_EQ(vec2.size(), 8);
  EXPECT_EQ(vec2.capacity(), 8);
  EXPECT_EQ(*iter, b);
  EXPECT_EQ(vec2[0], b1);
  EXPECT_EQ(vec2[1], b);
  EXPECT_EQ(vec2[2], b);
  EXPECT_EQ(vec2[3], b);
  EXPECT_EQ(vec2[4], b);
  EXPECT_EQ(vec2[5], b2);
  EXPECT_EQ(vec2[6], b3);
  EXPECT_EQ(vec2[7], b4);
}

TEST_F(SmallVectorUt, InsertListOk) {
  SmallVector<int64_t, 10> vec1;
  vec1.insert(vec1.end(), {1, 2, 3, 4, 5});
  int64_t expect1[] = {1,2,3,4,5};
  EXPECT_EQ(vec1.size(), 5);
  EXPECT_EQ(memcmp(vec1.data(), expect1, sizeof(expect1)), 0);

  vec1.insert(vec1.begin(), {10, 20, 30});
  int64_t expect2[] = {10,20,30,1,2,3,4,5};
  EXPECT_EQ(vec1.size(), 8);
  EXPECT_EQ(memcmp(vec1.data(), expect2, sizeof(expect2)), 0);

  // expand
  vec1.insert(vec1.begin() + 3, {100, 20, 30});
  int64_t expect3[] = {10,20,30,100,20,30,1,2,3,4,5};
  EXPECT_EQ(vec1.size(), 11);
  EXPECT_GE(vec1.capacity(), 11);
  EXPECT_EQ(memcmp(vec1.data(), expect3, sizeof(expect3)), 0);
}

TEST_F(SmallVectorUt, EmplaceOk) {
  SmallVector<std::vector<int64_t>, 8> vec2(4);

  std::vector<int64_t> b_1{1, 1, 1, 1, 1};
  std::vector<int64_t> b1{6, 7, 8, 9, 10};
  std::vector<int64_t> b2{11, 12, 13, 14, 15};
  std::vector<int64_t> b3{16, 17, 18, 19, 20};
  std::vector<int64_t> b4{21, 22, 13, 14, 15};
  vec2[0] = b1;
  vec2[1] = b2;
  vec2[2] = b3;
  vec2[3] = b4;
  auto iter = vec2.emplace(vec2.begin() + 1, 5, 1);  // b1, b[4], b2
  EXPECT_EQ(vec2.size(), 5);
  EXPECT_EQ(vec2.capacity(), 8);
  EXPECT_EQ(*iter, b_1);
  EXPECT_EQ(vec2[0], b1);
  EXPECT_EQ(vec2[1], b_1);
  EXPECT_EQ(vec2[2], b2);
  EXPECT_EQ(vec2[3], b3);
  EXPECT_EQ(vec2[4], b4);
}

TEST_F(SmallVectorUt, EraseOk) {
  SmallVector<int64_t, 10> vec1{1,2,3,4,5,6,7};

  vec1.erase(vec1.begin());
  EXPECT_EQ(vec1.size(), 6);
  int64_t vec1_expect_1[] = {2,3,4,5,6,7};
  EXPECT_EQ(memcmp(vec1.data(), vec1_expect_1, sizeof(vec1_expect_1)), 0);

  vec1.erase(vec1.begin() + 2);
  EXPECT_EQ(vec1.size(), 5);
  int64_t vec1_expect_2[] = {2,3,5,6,7};
  EXPECT_EQ(memcmp(vec1.data(), vec1_expect_2, sizeof(vec1_expect_2)), 0);
}

TEST_F(SmallVectorUt, EraseAllOk) {
  SmallVector<int64_t, 10> vec1{1,2,3,4,5,6,7};

  vec1.erase(vec1.begin(), vec1.end());
  EXPECT_EQ(vec1.size(), 0);
}

TEST_F(SmallVectorUt, EraseEmptyOk) {
  SmallVector<int64_t, 10> vec1{1,2,3,4,5,6,7};

  vec1.erase(vec1.begin(), vec1.begin());
  EXPECT_EQ(vec1.size(), 7);
  int64_t vec1_expect_1[] = {1,2,3,4,5,6,7};
  EXPECT_EQ(memcmp(vec1.data(), vec1_expect_1, sizeof(vec1_expect_1)), 0);
}

TEST_F(SmallVectorUt, Erase_CallCorrectConstructor) {
  SmallVector<FuncCounter, 10> vec1(11);

  FuncCounter::Clear();
  vec1.erase(vec1.begin());
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 11);
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 10);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(SmallVectorUt, PopBackOk) {
  SmallVector<int64_t, 10> vec1{1,2,3,4,5,6,7};

  vec1.pop_back();
  EXPECT_EQ(vec1.size(), 6);
  int64_t expect[] = {1,2,3,4,5,6};
  EXPECT_EQ(memcmp(vec1.data(), expect, sizeof(expect)), 0);
}

TEST_F(SmallVectorUt, PopBack_CallDestructor) {
  SmallVector<FuncCounter, 10> vec1(7);

  FuncCounter::Clear();
  vec1.pop_back();
  EXPECT_EQ(vec1.size(), 6);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 1);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(SmallVectorUt, ResizeOk1) {
  SmallVector<int64_t, 10> vec1{1,2,3,4,5,6,7};

  vec1.resize(9);
  int64_t expect_1[] = {1,2,3,4,5,6,7,0,0};
  EXPECT_EQ(vec1.size(), 9);
  EXPECT_EQ(memcmp(vec1.data(), expect_1, sizeof(expect_1)), 0);

  // expand
  vec1.resize(11);
  int64_t expect_2[] = {1,2,3,4,5,6,7,0,0,0,0};
  EXPECT_EQ(vec1.size(), 11);
  EXPECT_EQ(memcmp(vec1.data(), expect_2, sizeof(expect_2)), 0);

  // expand again
  auto next_size = vec1.capacity() + 1;
  vec1.resize(next_size);
  auto expect_3 = std::unique_ptr<int64_t[]>(new int64_t[next_size]());
  for (int64_t i = 0; i < 7; ++i) {
    expect_3[i] = i + 1;
  }
  EXPECT_EQ(vec1.size(), next_size);
  EXPECT_EQ(memcmp(vec1.data(), expect_3.get(), sizeof(int64_t) * next_size), 0);
}

TEST_F(SmallVectorUt, ResizeOk2) {
  SmallVector<int64_t, 10> vec1{1,2,3,4,5,6,7};

  vec1.resize(5);
  int64_t expect_1[] = {1,2,3,4,5};
  EXPECT_EQ(vec1.size(), 5);
  EXPECT_EQ(memcmp(vec1.data(), expect_1, sizeof(expect_1)), 0);

  vec1.resize(7);
  int64_t expect_2[] = {1,2,3,4,5,0,0};
  EXPECT_EQ(vec1.size(), 7);
  EXPECT_EQ(memcmp(vec1.data(), expect_2, sizeof(expect_2)), 0);
}

TEST_F(SmallVectorUt, ResizeOk1_CallCorrectConstructor) {
  SmallVector<FuncCounter, 10> vec1(7);

  FuncCounter::Clear();
  vec1.resize(9);
  EXPECT_EQ(FuncCounter::GetClearConstructTimes(), 2);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  // expand
  vec1.resize(11);
  EXPECT_EQ(FuncCounter::GetClearConstructTimes(), 2);
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 9);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 9);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  // expand again
  auto next_size = vec1.capacity() + 1;
  vec1.resize(next_size);
  auto expect_3 = std::unique_ptr<int64_t[]>(new int64_t[next_size]());
  for (int64_t i = 0; i < 7; ++i) {
    expect_3[i] = i + 1;
  }
  EXPECT_EQ(FuncCounter::GetClearConstructTimes(), next_size - 11);
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 11);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 11);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(SmallVectorUt, PushBackOk1) {
  SmallVector<int64_t, 5> vec1({1,2,3});
  vec1.push_back(10);
  EXPECT_EQ(vec1.size(), 4);
  int64_t expect_1[] = {1,2,3,10};
  EXPECT_EQ(memcmp(vec1.data(), expect_1, sizeof(expect_1)), 0);
}

TEST_F(SmallVectorUt, PushBackExpandOk2) {
  SmallVector<int64_t, 5> vec1({1,2,3,4,5});
  vec1.push_back(10);
  EXPECT_EQ(vec1.size(), 6);
  int64_t expect_1[] = {1,2,3,4,5,10};
  EXPECT_EQ(memcmp(vec1.data(), expect_1, sizeof(expect_1)), 0);
}

TEST_F(SmallVectorUt, PushBackExpandOk3) {
  SmallVector<int64_t, 5> vec1({1,2,3,4,5,6});
  vec1.push_back(10);
  EXPECT_EQ(vec1.size(), 7);
  int64_t expect_1[] = {1,2,3,4,5,6,10};
  EXPECT_EQ(memcmp(vec1.data(), expect_1, sizeof(expect_1)), 0);
}

TEST_F(SmallVectorUt, PushBack_CallCorrectConstructor) {
  SmallVector<FuncCounter, 5> vec1(4);
  FuncCounter fc;

  FuncCounter::Clear();
  vec1.push_back(fc);
  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 1);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  FuncCounter::Clear();
  vec1.push_back(fc);
  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 1);
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 5);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 5);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

    FuncCounter::Clear();
    vec1.push_back(std::move(fc));
    EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 1);
    EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(SmallVectorUt, SwapOk1) {
  SmallVector<int32_t, 10> vec1{1,2,3,4,5000};
  SmallVector<int32_t, 10> vec2{6,7,8,9000};

  vec1.swap(vec2);

  EXPECT_EQ(vec1.size(), 4);
  int32_t expect_2[] = {6,7,8,9000};
  EXPECT_EQ(memcmp(vec1.data(), expect_2, sizeof(expect_2)), 0);

  EXPECT_EQ(vec2.size(), 5);
  int32_t expect_1[] = {1,2,3,4,5000};
  EXPECT_EQ(memcmp(vec2.data(), expect_1, sizeof(expect_1)), 0);
}

TEST_F(SmallVectorUt, SwapOk2) {
  SmallVector<int32_t, 2> vec1{1,2,3,4,5000};
  SmallVector<int32_t, 2> vec2{6,7,8,9000};

  vec1.swap(vec2);

  EXPECT_EQ(vec1.size(), 4);
  int32_t expect_2[] = {6,7,8,9000};
  EXPECT_EQ(memcmp(vec1.data(), expect_2, sizeof(expect_2)), 0);

  EXPECT_EQ(vec2.size(), 5);
  int32_t expect_1[] = {1,2,3,4,5000};
  EXPECT_EQ(memcmp(vec2.data(), expect_1, sizeof(expect_1)), 0);
}

TEST_F(SmallVectorUt, SwapOk3) {
  SmallVector<int32_t, 4> vec1{1,2,3,4,5000};
  SmallVector<int32_t, 4> vec2{6,7,8,9000};

  vec1.swap(vec2);

  EXPECT_EQ(vec1.size(), 4);
  int32_t expect_2[] = {6,7,8,9000};
  EXPECT_EQ(memcmp(vec1.data(), expect_2, sizeof(expect_2)), 0);

  EXPECT_EQ(vec2.size(), 5);
  int32_t expect_1[] = {1,2,3,4,5000};
  EXPECT_EQ(memcmp(vec2.data(), expect_1, sizeof(expect_1)), 0);
}

TEST_F(SmallVectorUt, SwapOk3_ConstructTimes) {
  SmallVector<FuncCounter, 4> vec1(5);
  SmallVector<FuncCounter, 4> vec2(4);

  FuncCounter::Clear();
  vec1.swap(vec2);
  // vec1的指针被转移到vec2，vec1的对象没有操作
  // vec2的对象移动到vec1，vec2的原对象析构
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 4);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 4);
  EXPECT_TRUE(FuncCounter::AllTimesZero());

  std::swap(vec1, vec2);
  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 4);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 4);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(SmallVectorUt, CompareOperator) {
  SmallVector<int32_t, 4> vec1({1,2,3,4});
  SmallVector<int32_t, 4> vec2({1,2,3,4});
  SmallVector<int32_t, 5> vec3({1,2,3,4,5});
  SmallVector<int32_t, 5> vec4({1,2,3,5});
  SmallVector<int32_t, 4> vec5({1,2,3,5});
  SmallVector<int32_t, 4> vec6({1,2,3,4,5});

  EXPECT_TRUE(vec1 == vec2);
  EXPECT_FALSE(vec1 == vec3);
  EXPECT_FALSE(vec1 == vec4);
  EXPECT_FALSE(vec1 == vec5);
  EXPECT_FALSE(vec1 == vec6);
  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec1 != vec2);
  EXPECT_TRUE(vec1 != vec3);
  EXPECT_FALSE(vec1 == vec3);

  EXPECT_TRUE(vec1 < vec3);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 < vec4);
  EXPECT_TRUE(vec3 <= vec4);
}

TEST_F(SmallVectorUt, ReserveOk1) {
  SmallVector<int32_t, 4> vec1;
  EXPECT_EQ(vec1.size(), 0);
  EXPECT_EQ(vec1.capacity(), 4);

  vec1.reserve(3);
  EXPECT_EQ(vec1.size(), 0);
  EXPECT_EQ(vec1.capacity(), 4);

  vec1.reserve(5);
  EXPECT_EQ(vec1.size(), 0);
  EXPECT_GE(vec1.capacity(), 5);
}

TEST_F(SmallVectorUt, ReserveOk2) {
  SmallVector<int32_t, 4> vec1{1,2,3};
  EXPECT_EQ(vec1.size(), 3);
  EXPECT_EQ(vec1.capacity(), 4);

  vec1.reserve(3);
  EXPECT_EQ(vec1.size(), 3);
  EXPECT_EQ(vec1.capacity(), 4);

  vec1.reserve(5);
  EXPECT_EQ(vec1.size(), 3);
  EXPECT_GE(vec1.capacity(), 5);
  EXPECT_EQ(vec1[0], 1);
  EXPECT_EQ(vec1[1], 2);
  EXPECT_EQ(vec1[2], 3);
}
}  // namespace ge