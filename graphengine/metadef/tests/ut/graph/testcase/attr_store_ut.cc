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
#include "graph/attr_store.h"

namespace ge {
namespace {
struct TestStructB {
  int16_t a;
  int b;
  int64_t c;
  bool operator==(const TestStructB &other) const {
    return a == other.a && b == other.b && c == other.c;
  }
};
}
class AttrStoreUt : public testing::Test {};

TEST_F(AttrStoreUt, CreateAndGetOk1) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<bool>(0, true));
  EXPECT_TRUE(s.Set<bool>(1, false));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_TRUE(*s.Get<bool>(0));
  EXPECT_NE(s.Get<bool>(1), nullptr);
  EXPECT_FALSE(*s.Get<bool>(1));

  EXPECT_NE(s.GetByName<bool>("transpose_x1"), nullptr);
  EXPECT_TRUE(*s.GetByName<bool>("transpose_x1"));
  EXPECT_NE(s.GetByName<bool>("transpose_x2"), nullptr);
  EXPECT_FALSE(*s.GetByName<bool>("transpose_x2"));
}

TEST_F(AttrStoreUt, CreateAndGetOk2) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set(0, true));
  EXPECT_TRUE(s.Set(1, false));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_TRUE(*s.Get<bool>(0));
  EXPECT_NE(s.Get<bool>(1), nullptr);
  EXPECT_FALSE(*s.Get<bool>(1));
}

TEST_F(AttrStoreUt, CreateAndGetOk3) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set(0, true));
  EXPECT_TRUE(s.Set(1, TestStructB({1,2,3})));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_TRUE(*s.Get<bool>(0));
  EXPECT_NE(s.Get<TestStructB>(1), nullptr);
  EXPECT_EQ(*s.Get<TestStructB>(1), TestStructB({1,2,3}));

  EXPECT_NE(s.GetByName<bool>("transpose_x1"), nullptr);
  EXPECT_TRUE(*s.GetByName<bool>("transpose_x1"));
  EXPECT_NE(s.GetByName<TestStructB>("transpose_x2"), nullptr);
  EXPECT_EQ(*s.GetByName<TestStructB>("transpose_x2"), TestStructB({1,2,3}));
}

TEST_F(AttrStoreUt, CreateAndGetOk_RLValue1) {
  int a = 10;
  int &b = a;
  const int c = 20;

  auto s = AttrStore::Create(4);
  EXPECT_TRUE(s.Set(0, a));
  EXPECT_TRUE(s.Set(1, b));
  EXPECT_TRUE(s.Set(2, c));
  EXPECT_TRUE(s.Set(3, 20));

  EXPECT_EQ(*s.Get<int>(0), 10);
  EXPECT_EQ(*s.Get<int>(1), 10);
  EXPECT_EQ(*s.Get<int>(2), 20);
  EXPECT_EQ(*s.Get<int>(3), 20);
}

TEST_F(AttrStoreUt, CreateAndGetOk_RLValue2) {
  TestStructB a = {10, 20, 30};
  TestStructB &b = a;
  const TestStructB c = {100, 200, 300};

  auto s = AttrStore::Create(4);
  EXPECT_TRUE(s.SetByName("attr_0", a));
  EXPECT_TRUE(s.SetByName("attr_1", b));
  EXPECT_TRUE(s.SetByName("attr_2", c));
  EXPECT_TRUE(s.SetByName("attr_3", TestStructB{100,200,300}));

  EXPECT_EQ(*s.GetByName<TestStructB>("attr_0"), a);
  EXPECT_EQ(*s.GetByName<TestStructB>("attr_1"), a);
  EXPECT_EQ(*s.GetByName<TestStructB>("attr_2"), c);
  EXPECT_EQ(*s.GetByName<TestStructB>("attr_3"), c);
}

TEST_F(AttrStoreUt, CreateAndGetByNameOk1) {
  auto s = AttrStore::Create(2);

  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_TRUE(s.SetByName("transpose_x1", true));
  EXPECT_TRUE(s.SetByName("transpose_x2", false));

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_TRUE(*s.Get<bool>(0));
  EXPECT_NE(s.Get<bool>(1), nullptr);
  EXPECT_FALSE(*s.Get<bool>(1));

  EXPECT_NE(s.GetByName<bool>("transpose_x1"), nullptr);
  EXPECT_TRUE(*s.GetByName<bool>("transpose_x1"));
  EXPECT_NE(s.GetByName<bool>("transpose_x2"), nullptr);
  EXPECT_FALSE(*s.GetByName<bool>("transpose_x2"));
}

TEST_F(AttrStoreUt, CreateAndGetByNameOk2) {
  auto s = AttrStore::Create(2);

  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_TRUE(s.SetByName("transpose_x3", true));
  EXPECT_TRUE(s.SetByName("transpose_x4", false));

  EXPECT_EQ(s.Get<bool>(0), nullptr);
  EXPECT_EQ(s.Get<bool>(1), nullptr);

  EXPECT_NE(s.GetByName<bool>("transpose_x3"), nullptr);
  EXPECT_NE(s.GetByName<bool>("transpose_x4"), nullptr);

  EXPECT_EQ(*s.GetByName<bool>("transpose_x3"), true);
  EXPECT_EQ(*s.GetByName<bool>("transpose_x4"), false);
}

TEST_F(AttrStoreUt, GetNotExists) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<bool>(0, true));
  EXPECT_TRUE(s.Set<bool>(1, false));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_EQ(s.Get<bool>(2), nullptr);
  EXPECT_EQ(s.MutableGet<bool>(2), nullptr);

  EXPECT_EQ(s.GetByName<bool>("transpose_x3"), nullptr);
  EXPECT_EQ(s.MutableGetByName<bool>("transpose_x3"), nullptr);
}

TEST_F(AttrStoreUt, DeleteOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<bool>(0, true));
  EXPECT_TRUE(s.Set<bool>(1, false));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_NE(s.Get<bool>(1), nullptr);

  EXPECT_TRUE(s.Delete("transpose_x1"));
  EXPECT_EQ(s.Get<bool>(0), nullptr);
  EXPECT_FALSE(s.Delete("transpose_x1"));
}

TEST_F(AttrStoreUt, GetWithWrongType) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<bool>(0, true));
  EXPECT_TRUE(s.Set<TestStructB>(1, {1,2,10}));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_NE(s.Get<TestStructB>(1), nullptr);
  EXPECT_EQ(s.Get<int>(0), nullptr);
  EXPECT_EQ(s.Get<int>(1), nullptr);
}

TEST_F(AttrStoreUt, ModifyOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<bool>(0, true));
  EXPECT_TRUE(s.Set<bool>(1, false));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_TRUE(*s.Get<bool>(0));
  EXPECT_FALSE(*s.Get<bool>(1));

  EXPECT_TRUE(s.Set<bool>(0, false));
  EXPECT_FALSE(*s.Get<bool>(0));
}

TEST_F(AttrStoreUt, ModifyByNameOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<int64_t>(1, 200));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  auto p = s.MutableGetByName<int64_t>("transpose_x1");
  EXPECT_NE(p,  nullptr);
  *p = 101;
  EXPECT_EQ(*s.Get<int64_t>(0), 101);
  EXPECT_EQ(*s.GetByName<int64_t>("transpose_x1"), 101);


  p = s.MutableGetByName<int64_t>("transpose_x2");
  EXPECT_NE(p,  nullptr);
  *p = 201;
  EXPECT_EQ(*s.Get<int64_t>(1), 201);
  EXPECT_EQ(*s.GetByName<int64_t>("transpose_x2"), 201);
}

TEST_F(AttrStoreUt, ExistsOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<int64_t>(1, 200));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_TRUE(s.Exists(0));
  EXPECT_TRUE(s.Exists(1));
  EXPECT_TRUE(s.Exists("transpose_x1"));
  EXPECT_TRUE(s.Exists("transpose_x2"));
  EXPECT_FALSE(s.Exists(2));
  EXPECT_FALSE(s.Exists("transpose_x3"));
}

TEST_F(AttrStoreUt, CopyOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<bool>(1, true));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);
  s.SetByName("attr_3", TestStructB{10,200,3000});
  s.SetByName("attr_4", std::vector<int64_t>({1,2,3,4,5}));

  auto s2(s);

  EXPECT_NE(s2.Get<int64_t>(0), nullptr);
  EXPECT_NE(s2.Get<int64_t>(0), s.Get<int64_t>(0));
  EXPECT_EQ(*s2.Get<int64_t>(0), 100);

  EXPECT_NE(s2.Get<bool>(1), nullptr);
  EXPECT_NE(s2.Get<bool>(1), s.Get<bool>(1));
  EXPECT_EQ(*s2.Get<bool>(1), true);

  EXPECT_NE(s2.GetByName<int64_t>("transpose_x1"), nullptr);
  EXPECT_NE(s2.GetByName<int64_t>("transpose_x1"), s.GetByName<int64_t>("transpose_x1"));
  EXPECT_EQ(*s2.GetByName<int64_t>("transpose_x1"), 100);

  EXPECT_NE(s2.GetByName<bool>("transpose_x2"), nullptr);
  EXPECT_NE(s2.GetByName<bool>("transpose_x2"), s.GetByName<bool>("transpose_x2"));
  EXPECT_EQ(*s2.GetByName<bool>("transpose_x2"), true);

  EXPECT_NE(s2.GetByName<TestStructB>("attr_3"), nullptr);
  EXPECT_NE(s2.GetByName<TestStructB>("attr_3"), s.GetByName<TestStructB>("attr_3"));
  EXPECT_EQ(*s2.GetByName<TestStructB>("attr_3"), TestStructB({10,200,3000}));

  EXPECT_NE(s2.GetByName<std::vector<int64_t>>("attr_4"), nullptr);
  EXPECT_NE(s2.GetByName<std::vector<int64_t>>("attr_4"), s.GetByName<std::vector<int64_t>>("attr_4"));
  EXPECT_EQ(*s2.GetByName<std::vector<int64_t>>("attr_4"), std::vector<int64_t>({1,2,3,4,5}));
}

TEST_F(AttrStoreUt, MoveOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<bool>(1, true));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);
  s.SetByName("attr_3", TestStructB{10,200,3000});
  s.SetByName("attr_4", std::vector<int64_t>({1,2,3,4,5}));

  auto attr_3 = s.GetByName<TestStructB>("attr_3");
  auto attr_4 = s.GetByName<std::vector<int64_t>>("attr_4");

  auto s2(std::move(s));

  EXPECT_NE(s2.GetByName<TestStructB>("attr_3"), nullptr);
  EXPECT_EQ(s2.GetByName<TestStructB>("attr_3"), attr_3);
  EXPECT_EQ(*s2.GetByName<TestStructB>("attr_3"), TestStructB({10,200,3000}));

  EXPECT_NE(s2.GetByName<std::vector<int64_t>>("attr_4"), nullptr);
  EXPECT_EQ(s2.GetByName<std::vector<int64_t>>("attr_4"), attr_4);
  EXPECT_EQ(*s2.GetByName<std::vector<int64_t>>("attr_4"), std::vector<int64_t>({1,2,3,4,5}));
}

TEST_F(AttrStoreUt, GetAllAttrNamesOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<bool>(1, true));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);
  s.SetByName("attr_3", TestStructB{10,200,3000});
  s.SetByName("attr_4", std::vector<int64_t>({1,2,3,4,5}));

  EXPECT_EQ(s.GetAllAttrNames(), std::set<std::string>({"transpose_x1",
                                                        "transpose_x2",
                                                        "attr_3",
                                                        "attr_4"}));
}

TEST_F(AttrStoreUt, GetAllAttrsOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<bool>(1, true));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);
  s.SetByName("attr_3", TestStructB{10,200,3000});
  s.SetByName("attr_4", std::vector<int64_t>({1,2,3,4,5}));

  auto attrs = s.GetAllAttrs();
  EXPECT_EQ(attrs.size(), 4);
  EXPECT_EQ(*attrs["transpose_x1"].Get<int64_t>(), 100);
  EXPECT_EQ(*attrs["transpose_x2"].Get<bool>(), true);
  EXPECT_EQ(*attrs["attr_3"].Get<TestStructB>(), TestStructB({10,200,3000}));
  EXPECT_EQ(*attrs["attr_4"].Get<std::vector<int64_t>>(), std::vector<int64_t>({1,2,3,4,5}));
}

TEST_F(AttrStoreUt, GetAllAttrs_EmptyPredefinedAttrsNotReturn) {
  auto s = AttrStore::Create(2);
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  s.SetByName("attr_3", TestStructB{10,200,3000});
  s.SetByName("attr_4", std::vector<int64_t>({1,2,3,4,5}));

  auto attrs = s.GetAllAttrs();
  EXPECT_EQ(attrs.size(), 3);
  EXPECT_EQ(attrs.count("transpose_x2"), 0);
}
}