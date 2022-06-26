/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "ut/tools/converter/parser/tflite/tflite_parsers_test_utils.h"
#include <iostream>
#include "common/common_test.h"

namespace mindspore {
class TestTfliteLogicalParserAnd : public TestTfliteParser {
 public:
  TestTfliteLogicalParserAnd() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./logical_and.tflite", ""); }
};

TEST_F(TestTfliteLogicalParserAnd, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_LogicalAnd) << "wrong Op Type";
}

class TestTfliteParserLogicalNot : public TestTfliteParser {
 public:
  TestTfliteParserLogicalNot() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./logical_not.tflite", ""); }
};

TEST_F(TestTfliteParserLogicalNot, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_LogicalNot) << "wrong Op Type";
}

class TestTfliteParserLogicalOr : public TestTfliteParser {
 public:
  TestTfliteParserLogicalOr() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./logical_or.tflite", ""); }
};

TEST_F(TestTfliteParserLogicalOr, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_LogicalOr) << "wrong Op Type";
}

}  // namespace mindspore
