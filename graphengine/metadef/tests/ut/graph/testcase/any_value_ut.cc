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
#include <iostream>
#include "test_structs.h"
#include "func_counter.h"
#include "graph/any_value.h"
#include "graph/ge_attr_value.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "graph/buffer.h"
namespace ge {
namespace {
struct InlineFuncCounter : public FuncCounter {
  int32_t a;
};
struct AllocatedFuncCounter : public FuncCounter {
  int64_t a;
  int64_t b;
  int64_t c;
  int64_t d;
};
}

class AnyValueUt : public testing::Test {};

TEST_F(AnyValueUt, InlineClassBuildOk1) {
  InlineStructB s1;
  InlineStructB &s2 = s1;
  const InlineStructB s3;

  auto av1 = AnyValue::CreateFrom(s1);
  auto av2 = AnyValue::CreateFrom(s2);
  auto av3 = AnyValue::CreateFrom(s3);
  auto av4 = AnyValue::CreateFrom(InlineStructB());

  auto av2_1 = AnyValue::CreateFrom<InlineStructB>(s1);
  auto av2_2 = AnyValue::CreateFrom<InlineStructB>(s2);
  auto av2_3 = AnyValue::CreateFrom<InlineStructB>(s3);
  auto av2_4 = AnyValue::CreateFrom<InlineStructB>(InlineStructB());
}
TEST_F(AnyValueUt, BasicTypesBuildOk1) {
  int i1{10};
  int &i2 = i1;
  const int i3 = 20;

  auto av1 = AnyValue::CreateFrom(i1);
  auto av2 = AnyValue::CreateFrom(i2);
  auto av3 = AnyValue::CreateFrom(i3);
  auto av4 = AnyValue::CreateFrom(30);

  auto av2_1 = AnyValue::CreateFrom<int>(i1);
  auto av2_2 = AnyValue::CreateFrom<int>(i2);
  auto av2_3 = AnyValue::CreateFrom<int>(i3);
  auto av2_4 = AnyValue::CreateFrom<int>(30);

  // todo 验证av1~4为inline保存
}

TEST_F(AnyValueUt, CopyConstructOk_Inline1) {
  AnyValue av;
  InlineFuncCounter fc;
  av.SetValue(fc);

  FuncCounter::Clear();
  AnyValue av2(av);

  EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 1);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(AnyValueUt, CopyConstructOk_Alloc1) {
  AnyValue av;
  AllocatedFuncCounter fc;
  av.SetValue(fc);

  FuncCounter::Clear();
  AnyValue av2(av);

    EXPECT_EQ(FuncCounter::GetClearCopyConstructTimes(), 1);
    EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(AnyValueUt, CopyConstructOk) {
  AnyValue av;
  std::string s("Hello world");
  av.SetValue(s);

  AnyValue av2(av);
  EXPECT_EQ(*av2.Get<std::string>(), "Hello world");
  s = "asdfa";
  EXPECT_EQ(*av2.Get<std::string>(), "Hello world");

}

TEST_F(AnyValueUt, CopyAssignOk) {
  AnyValue av;
  std::string s("Hello world");
  av.SetValue(s);

  AnyValue av2;
  av2 = av;
  EXPECT_EQ(*av2.Get<std::string>(), "Hello world");
  s = "asdfa";
  EXPECT_EQ(*av2.Get<std::string>(), "Hello world");
}

TEST_F(AnyValueUt, MoveConstructOk_Inline1) {
  {
    AnyValue av;
    {
      InlineFuncCounter fc;
      av.SetValue(fc);
    }

    FuncCounter::Clear();
    AnyValue av2(std::move(av));
  }

  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 1);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 2);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}
TEST_F(AnyValueUt, MoveConstructOk_Alloc1) {
  {
    AnyValue av;
    {
      AllocatedFuncCounter fc;
      av.SetValue(fc);
    }

    FuncCounter::Clear();
    AnyValue av2(std::move(av));
  }
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 1);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(AnyValueUt, MoveAssign_Inline1) {
  {
    AnyValue av;
    AnyValue av2;
    {
      InlineFuncCounter fc;
      av.SetValue(fc);
    }

    FuncCounter::Clear();
    av2 = std::move(av);
  }


  EXPECT_EQ(FuncCounter::GetClearMoveConstructTimes(), 1);
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 2);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}
TEST_F(AnyValueUt, MoveAssignOk_Alloc1) {
  {
    AnyValue av;
    AnyValue av2;
    {
      AllocatedFuncCounter fc;
      av.SetValue(fc);
    }

    FuncCounter::Clear();
    av2 = std::move(av);
  }
  EXPECT_EQ(FuncCounter::GetClearDestructTimes(), 1);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(AnyValueUt, ClassBuildOk1) {
  TestStructA a1 = {0, 10, 2000};
  TestStructA &a2 = a1;
  const TestStructA a3 = {0, 100, 3000};

  auto av1 = AnyValue::CreateFrom(a1);
  auto av2 = AnyValue::CreateFrom(a2);
  auto av3 = AnyValue::CreateFrom(a3);
  auto av4 = AnyValue::CreateFrom(TestStructA{0, 100, 3000});

  auto av2_1 = AnyValue::CreateFrom<TestStructA>(a1);
  auto av2_2 = AnyValue::CreateFrom<TestStructA>(a2);
  auto av2_3 = AnyValue::CreateFrom<TestStructA>(a3);
  auto av2_4 = AnyValue::CreateFrom<TestStructA>(TestStructA{0, 100, 3000});

  // todo 验证av1~4为allocate保存
  // todo 验证av1~3走了拷贝构造，av4走了移动构造
}

TEST_F(AnyValueUt, ClearIsOk) {
  auto av1 = AnyValue::CreateFrom(TestStructA{0, 100, 3000});
  auto av2 = AnyValue::CreateFrom(30);

  EXPECT_FALSE(av1.IsEmpty());
  av1.Clear();
  EXPECT_TRUE(av1.IsEmpty());

  EXPECT_FALSE(av2.IsEmpty());
  av2.Clear();
  EXPECT_TRUE(av2.IsEmpty());
}

TEST_F(AnyValueUt, BasicTypeesSameTypeOk) {
  int i1{10};
  int &i2 = i1;
  const int i3 = 20;

  auto av1 = AnyValue::CreateFrom(i1);
  auto av2 = AnyValue::CreateFrom(i2);
  auto av3 = AnyValue::CreateFrom(i3);
  auto av4 = AnyValue::CreateFrom(30);

  EXPECT_TRUE(av1.SameType<int>());
  EXPECT_TRUE(av2.SameType<int>());
  EXPECT_TRUE(av3.SameType<int>());
  EXPECT_TRUE(av4.SameType<int>());

  EXPECT_FALSE(av1.SameType<float>());
  EXPECT_FALSE(av2.SameType<float>());
  EXPECT_FALSE(av3.SameType<float>());
  EXPECT_FALSE(av4.SameType<float>());
  EXPECT_FALSE(av1.SameType<TestStructA>());
  EXPECT_FALSE(av2.SameType<TestStructA>());
  EXPECT_FALSE(av3.SameType<TestStructA>());
  EXPECT_FALSE(av4.SameType<TestStructA>());
}

TEST_F(AnyValueUt, BasicTypesBuildAndGetOk1) {
  int i1{10};
  int &i2 = i1;
  const int i3 = 20;

  auto v1 = AnyValue::CreateFrom(i1);
  auto v2 = AnyValue::CreateFrom(i2);
  auto v3 = AnyValue::CreateFrom(i3);
  auto v4 = AnyValue::CreateFrom(30);

  auto v2_1 = AnyValue::CreateFrom<int>(i1);
  auto v2_2 = AnyValue::CreateFrom<int>(i2);
  auto v2_3 = AnyValue::CreateFrom<int>(i3);
  auto v2_4 = AnyValue::CreateFrom<int>(30);

  EXPECT_EQ(*v1.Get<int>(), 10);
  EXPECT_EQ(*v2.Get<int>(), 10);
  EXPECT_EQ(*v3.Get<int>(), 20);
  EXPECT_EQ(*v4.Get<int>(), 30);

  EXPECT_EQ(*v2_1.Get<int>(), 10);
  EXPECT_EQ(*v2_2.Get<int>(), 10);
  EXPECT_EQ(*v2_3.Get<int>(), 20);
  EXPECT_EQ(*v2_4.Get<int>(), 30);
}

TEST_F(AnyValueUt, BasicTypesBuildAndGetOk2) {
  bool i1{true};
  bool &i2 = i1;
  const bool i3 = true;
  bool i5{false};
  bool &i6 = i5;
  const bool i7 = false;

  auto v1 = AnyValue::CreateFrom(i1);
  auto v2 = AnyValue::CreateFrom(i2);
  auto v3 = AnyValue::CreateFrom(i3);
  auto v4 = AnyValue::CreateFrom(true);
  auto v5 = AnyValue::CreateFrom(i5);
  auto v6 = AnyValue::CreateFrom(i6);
  auto v7 = AnyValue::CreateFrom(i7);
  auto v8 = AnyValue::CreateFrom(false);
  EXPECT_EQ(*v1.Get<bool>(), true);
  EXPECT_EQ(*v2.Get<bool>(), true);
  EXPECT_EQ(*v3.Get<bool>(), true);
  EXPECT_EQ(*v4.Get<bool>(), true);
  EXPECT_EQ(*v5.Get<bool>(), false);
  EXPECT_EQ(*v6.Get<bool>(), false);
  EXPECT_EQ(*v7.Get<bool>(), false);
  EXPECT_EQ(*v8.Get<bool>(), false);
}

TEST_F(AnyValueUt, BasicTypesAssignOk) {
  AnyValue v1 = AnyValue::CreateFrom(10);
  EXPECT_EQ(*v1.Get<int>(), 10);

  v1.SetValue(true);
  EXPECT_EQ(*v1.Get<bool>(), true);
  v1.SetValue(200);
  EXPECT_EQ(*v1.Get<int>(), 200);
  v1.SetValue(false);
  EXPECT_EQ(*v1.Get<bool>(), false);
}

TEST_F(AnyValueUt, MutableBasicTypeOk) {
  AnyValue v1 = AnyValue::CreateFrom<int>(10);
  EXPECT_EQ(*v1.Get<int>(), 10);
  auto p = v1.MutableGet<int>();
  *p = 20;
  EXPECT_EQ(*v1.Get<int>(), 20);
}

TEST_F(AnyValueUt, ClassBuildAndGetOk1) {
  TestStructA a1 = {0, 10, 2000};
  TestStructA &a2 = a1;
  const TestStructA a3 = {0, 100, 3000};

  auto v1 = AnyValue::CreateFrom(a1);
  auto v2 = AnyValue::CreateFrom(a2);
  auto v3 = AnyValue::CreateFrom(a3);
  auto v4 = AnyValue::CreateFrom(TestStructA{0, 100, 3000});

  auto v2_1 = AnyValue::CreateFrom<TestStructA>(a1);
  auto v2_2 = AnyValue::CreateFrom<TestStructA>(a2);
  auto v2_3 = AnyValue::CreateFrom<TestStructA>(a3);
  auto v2_4 = AnyValue::CreateFrom<TestStructA>(TestStructA{0, 100, 3000});

  EXPECT_EQ(*v1.Get<TestStructA>(), a1);
  EXPECT_EQ(*v2.Get<TestStructA>(), a1);
  EXPECT_EQ(*v3.Get<TestStructA>(), a3);
  EXPECT_EQ(*v4.Get<TestStructA>(), a3);

  EXPECT_EQ(*v2_1.Get<TestStructA>(), a1);
  EXPECT_EQ(*v2_2.Get<TestStructA>(), a1);
  EXPECT_EQ(*v2_3.Get<TestStructA>(), a3);
  EXPECT_EQ(*v2_4.Get<TestStructA>(), a3);
}

TEST_F(AnyValueUt, ClassBuildAndGetOk2) {
  std::vector<int64_t> b1;
  b1.resize(100);
  for (int32_t i = 0; i < 100; ++i) {
    b1[i] = i * 10;
  }
  std::vector<int64_t> &b2 = b1;
  const std::vector<int64_t> b3{1, 2, 3, 4, 5};

  auto v1 = AnyValue::CreateFrom(b1);
  auto v2 = AnyValue::CreateFrom(b2);
  auto v3 = AnyValue::CreateFrom(b3);
  auto v4 = AnyValue::CreateFrom(std::vector<int64_t>({1, 2, 3, 4, 5}));
  auto v5 = AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3, 4, 5});

  EXPECT_EQ(*v1.Get<std::vector<int64_t>>(), b1);
  EXPECT_EQ(*v2.Get<std::vector<int64_t>>(), b1);
  EXPECT_EQ(*v3.Get<std::vector<int64_t>>(), b3);
  EXPECT_EQ(*v4.Get<std::vector<int64_t>>(), b3);
  EXPECT_EQ(*v5.Get<std::vector<int64_t>>(), b3);
}

TEST_F(AnyValueUt, SetAndGetOk1) {
  int i1{10};

  AnyValue v1;
  v1.SetValue<int>(i1);
  EXPECT_EQ(*v1.Get<int>(), 10);
  v1.SetValue<int>(20);
  EXPECT_EQ(*v1.Get<int>(), 20);
  v1.SetValue(false);
  EXPECT_EQ(*v1.Get<bool>(), false);
}

TEST_F(AnyValueUt, SetAndGetOk2) {
  const std::vector<int64_t> b3{1, 2, 3, 4, 5};

  AnyValue v1;
  v1.SetValue<int>(10);
  EXPECT_EQ(*v1.Get<int>(), 10);
  v1.SetValue(b3);
  EXPECT_EQ(*v1.Get<std::vector<int64_t>>(), b3);
  v1.SetValue(false);
  EXPECT_EQ(*v1.Get<bool>(), false);
}

TEST_F(AnyValueUt, SameTypeTestBasicTypes) {
  AnyValue v1;
  v1.SetValue<int32_t>(10);
  EXPECT_TRUE(v1.SameType<int32_t>());
  EXPECT_FALSE(v1.SameType<int64_t>());
  EXPECT_FALSE(v1.SameType<std::vector<int32_t>>());
}

TEST_F(AnyValueUt, SameTypeTestClass) {
  auto v1 = AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3});
  EXPECT_FALSE(v1.SameType<int32_t>());
  EXPECT_FALSE(v1.SameType<int64_t>());
  EXPECT_FALSE(v1.SameType<std::vector<int32_t>>());
  EXPECT_TRUE(v1.SameType<std::vector<int64_t>>());
}

TEST_F(AnyValueUt, GetAddrOk) {
  auto v1 = AnyValue::CreateFrom(10);

  auto value1 = v1.Get<int>();
  auto value2 = v1.MutableGet<int>();
  EXPECT_EQ(value1, value2);
}

TEST_F(AnyValueUt, MutableGetAndModifiedOk1) {
  auto v1 = AnyValue::CreateFrom(10);

  auto value = v1.MutableGet<int>();
  EXPECT_EQ(*value, 10);

  *value = 20;
  EXPECT_EQ(*v1.Get<int>(), 20);
}

TEST_F(AnyValueUt, MutableGetAndModifiedOk2) {
  auto v1 = AnyValue::CreateFrom(std::vector<int64_t>({1, 2, 3, 4}));

  auto value = v1.MutableGet<std::vector<int64_t>>();
  EXPECT_EQ(*value, std::vector<int64_t>({1, 2, 3, 4}));

  value->push_back(5);
  EXPECT_EQ(*v1.Get<std::vector<int64_t>>(), std::vector<int64_t>({1, 2, 3, 4, 5}));
}

TEST_F(AnyValueUt, MoveConstructOk1) {
  auto v1 = AnyValue::CreateFrom(std::vector<int64_t>({1, 2, 3, 4}));

  AnyValue v2(std::move(v1));
  EXPECT_EQ(*v2.Get<std::vector<int64_t>>(), std::vector<int64_t>({1, 2, 3, 4}));
}

TEST_F(AnyValueUt, SetRvalueOk) {
  AnyValue av;
  EXPECT_EQ(av.SetValue(std::vector<int64_t>({1,2,3,4})), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::vector<int64_t>>(), nullptr);
  EXPECT_EQ(*av.Get<std::vector<int64_t>>(), std::vector<int64_t>({1,2,3,4}));
}

TEST_F(AnyValueUt, SetGetValueOk) {
  AnyValue av;
  EXPECT_EQ(av.SetValue(std::vector<int64_t>({1,2,3,4})), GRAPH_SUCCESS);

  std::vector<int64_t> value;
  EXPECT_EQ(av.GetValue(value), GRAPH_SUCCESS);
  EXPECT_EQ(value, std::vector<int64_t>({1,2,3,4}));
}

TEST_F(AnyValueUt, SetGetValueOk_Inline) {
  AnyValue av;

  EXPECT_EQ(av.SetValue(InlineFuncCounter()), GRAPH_SUCCESS);
  InlineFuncCounter fc;

  FuncCounter::Clear();
  EXPECT_EQ(av.GetValue(fc), GRAPH_SUCCESS);
  EXPECT_EQ(FuncCounter::GetClearCopyAssignTimes(), 1);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(AnyValueUt, SetGetValueOk_Allocate) {
  AnyValue av;

  EXPECT_EQ(av.SetValue(AllocatedFuncCounter()), GRAPH_SUCCESS);
  AllocatedFuncCounter fc;

  FuncCounter::Clear();
  EXPECT_EQ(av.GetValue(fc), GRAPH_SUCCESS);
  EXPECT_EQ(FuncCounter::GetClearCopyAssignTimes(), 1);
  EXPECT_TRUE(FuncCounter::AllTimesZero());
}

TEST_F(AnyValueUt, GetWrongTypeFailed) {
  AnyValue av;
  av.SetValue(std::vector<int64_t>({1,2,3,4,5}));

  int64_t a;
  EXPECT_NE(av.GetValue(a), GRAPH_SUCCESS);
  EXPECT_EQ(av.Get<std::vector<int32_t>>(), nullptr);
}

TEST_F(AnyValueUt, GetEmptyOk) {
  AnyValue av;

  int64_t a;
  EXPECT_NE(av.GetValue(a), GRAPH_SUCCESS);
  EXPECT_EQ(av.Get<std::vector<int32_t>>(), nullptr);
}
TEST_F(AnyValueUt, SameTypeOk_Inline) {
  AnyValue av;
  av.SetValue(InlineFuncCounter());

  EXPECT_TRUE(av.SameType<InlineFuncCounter>());
  EXPECT_FALSE(av.SameType<std::vector<int32_t>>());
  EXPECT_FALSE(av.SameType<AllocatedFuncCounter>());
}

TEST_F(AnyValueUt, SameTypeOk_Allocate) {
  AnyValue av;
  av.SetValue(AllocatedFuncCounter());

  EXPECT_TRUE(av.SameType<AllocatedFuncCounter>());
  EXPECT_FALSE(av.SameType<std::vector<int32_t>>());
}

TEST_F(AnyValueUt, GetTypeOk) {
  AnyValue av;
  av.SetValue(std::string("abc"));
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::string>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_STRING);

  av.SetValue(static_cast<float>(1.0));
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<float>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_FLOAT);

  av.SetValue(true);
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<bool>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_BOOL);

  av.SetValue(static_cast<int64_t>(10));
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<int64_t>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_INT);

  av.SetValue(GeTensorDesc());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<GeTensorDesc>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_TENSOR_DESC);

  av.SetValue(GeTensor());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<GeTensor>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_TENSOR);

  av.SetValue(Buffer());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<Buffer>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_BYTES);

//  auto graph = proto::GraphDef(nullptr);
//  av.SetValue(graph);
//  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<proto::GraphDef>());
//  EXPECT_EQ(av.GetValueType(), AnyValue::VT_GRAPH);

  av.SetValue(NamedAttrs());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<NamedAttrs>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_NAMED_ATTRS);


  av.SetValue(std::vector<std::vector<int64_t>>({{1,2,3}, {1,2,3}}));
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<std::vector<int64_t>>>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_LIST_INT);

  av.SetValue(DT_FLOAT);
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<DataType>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_DATA_TYPE);

  av.SetValue(std::vector<std::vector<float>>({{1.0,2.1,3.3}, {1.2,2.4,3.5}}));
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<std::vector<float>>>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_LIST_FLOAT);

  av.SetValue(std::vector<std::string>({"abc", "cde"}));
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<std::string>>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_STRING);

  av.SetValue(std::vector<float>());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<float>>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_FLOAT);

  av.SetValue(std::vector<bool>());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<bool>>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_BOOL);

  av.SetValue(std::vector<int64_t>());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<int64_t>>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_INT);

  av.SetValue(std::vector<GeTensorDesc>());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<GeTensorDesc>>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_TENSOR_DESC);

  av.SetValue(std::vector<GeTensor>());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<GeTensor>>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_TENSOR);

  av.SetValue(std::vector<Buffer>());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<Buffer>>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_BYTES);

//  av.SetValue(std::vector<proto::GraphDef>());
//  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<proto::GraphDef>>());
//  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_GRAPH);

  av.SetValue(std::vector<NamedAttrs>());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<NamedAttrs>>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_NAMED_ATTRS);

  av.SetValue(std::vector<DataType>());
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<DataType>>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_LIST_DATA_TYPE);

    av.SetValue(std::vector<int32_t>());
    EXPECT_EQ(av.GetValueTypeId(), GetTypeId<std::vector<int32_t>>());
    EXPECT_EQ(av.GetValueType(), AnyValue::VT_NONE);
}

TEST_F(AnyValueUt, GetTypeOk_Inline) {
  AnyValue av;
  av.SetValue(true);
  EXPECT_EQ(av.GetValueTypeId(), GetTypeId<bool>());
  EXPECT_EQ(av.GetValueType(), AnyValue::VT_BOOL);
}

TEST_F(AnyValueUt, SwapOk1) {
  AnyValue av1;
  av1.SetValue(std::vector<int32_t>({1,2,3,4}));
  AnyValue av2;
  av2.SetValue(static_cast<int64_t>(1024));
  av1.Swap(av2);
  EXPECT_NE(av1.Get<int64_t>(), nullptr);
  EXPECT_EQ(*av1.Get<int64_t>(), 1024);

  EXPECT_NE(av2.Get<std::vector<int32_t>>(), nullptr);
  EXPECT_EQ(*av2.Get<std::vector<int32_t>>(), std::vector<int32_t>({1,2,3,4}));
}

TEST_F(AnyValueUt, SwapOk2) {
  AnyValue av1;
  av1.SetValue(std::vector<int32_t>({1,2,3,4}));
  AnyValue av2;
  av2.SetValue(TestStructA(10,11,12));
  av1.Swap(av2);
  EXPECT_NE(av1.Get<TestStructA>(), nullptr);
  EXPECT_EQ(*av1.Get<TestStructA>(), TestStructA(10,11,12));

  EXPECT_NE(av2.Get<std::vector<int32_t>>(), nullptr);
  EXPECT_EQ(*av2.Get<std::vector<int32_t>>(), std::vector<int32_t>({1,2,3,4}));
}

TEST_F(AnyValueUt, TestValueTypes) {
  std::set<AnyValue::ValueType> value_types = {
      AnyValue::VT_NONE,
      AnyValue::VT_STRING,
      AnyValue::VT_FLOAT,
      AnyValue::VT_BOOL,
      AnyValue::VT_INT,  // 5
      AnyValue::VT_TENSOR_DESC,
      AnyValue::VT_TENSOR,
      AnyValue::VT_BYTES,
      AnyValue::VT_GRAPH,
      AnyValue::VT_NAMED_ATTRS,  // 10
      AnyValue::VT_LIST_LIST_INT,
      AnyValue::VT_DATA_TYPE,
      AnyValue::VT_LIST_LIST_FLOAT,
      AnyValue::VT_LIST_STRING,
      AnyValue::VT_LIST_FLOAT,  // 15
      AnyValue::VT_LIST_BOOL,
      AnyValue::VT_LIST_INT,
      AnyValue::VT_LIST_TENSOR_DESC,
      AnyValue::VT_LIST_TENSOR,
      AnyValue::VT_LIST_BYTES,  // 20
      AnyValue::VT_LIST_GRAPH,
      AnyValue::VT_LIST_NAMED_ATTRS,
      AnyValue::VT_LIST_DATA_TYPE,
  };
  EXPECT_EQ(value_types.size(), 23);
}

TEST_F(AnyValueUt, ConstructFromEmtpy) {
  AnyValue av1;
  const AnyValue av2;
  AnyValue &av3 = av1;
  const AnyValue &av4 = av2;

  AnyValue tav1(av1);
  AnyValue tav2(av2);
  AnyValue tav3(av3);
  AnyValue tav4(av4);
  AnyValue tav5(AnyValue());
}

TEST_F(AnyValueUt, AssignFromEmtpy) {
  AnyValue av1;
  const AnyValue av2;
  AnyValue &av3 = av1;
  const AnyValue &av4 = av2;

  AnyValue tav1;
  AnyValue tav2;
  AnyValue tav3;
  AnyValue tav4;
  AnyValue tav5;

  tav1 = av1;
  tav2 = av2;
  tav3 = av3;
  tav4 = av4;
  tav5 = AnyValue();
}
TEST_F(AnyValueUt, SwapWithEmpty) {
  AnyValue av1;
  AnyValue av2 = AnyValue::CreateFrom<int64_t>(10);
  AnyValue av3;

  av1.Swap(av2);
  EXPECT_EQ(*av1.Get<int64_t>(), 10);
  EXPECT_TRUE(av2.IsEmpty());

  av1.Swap(av2);
  EXPECT_TRUE(av1.IsEmpty());
  EXPECT_EQ(*av2.Get<int64_t>(), 10);
}
}  // namespace ge