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
#include "graph/utils/dumper/ge_graph_dumper.h"
#include <gtest/gtest.h>

namespace ge {
class GeGraphDumperUt : public testing::Test {};

namespace {
int64_t dump_times = 0;
class TestDumper : public GeGraphDumper {
 public:
  void Dump(const ComputeGraphPtr &graph, const string &suffix) override {
    ++dump_times;
  }
};
}

TEST_F(GeGraphDumperUt, DefaultImpl)  {
  dump_times = 0;
  GraphDumperRegistry::Unregister();
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 0);
}

TEST_F(GeGraphDumperUt, RegisterOk)  {
  dump_times = 0;
  TestDumper dumper;
  GraphDumperRegistry::Unregister();
  GraphDumperRegistry::Register(dumper);
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 1);
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 2);
}

TEST_F(GeGraphDumperUt, UnregisterOk)  {
  dump_times = 0;
  TestDumper dumper;
  GraphDumperRegistry::Register(dumper);
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 1);
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 2);

  GraphDumperRegistry::Unregister();
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 2);
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 2);
}
}
