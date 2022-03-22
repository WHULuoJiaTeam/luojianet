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
#include "graph/ge_local_context.h"

namespace ge {
class UtestGeLocalContext : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGeLocalContext, GetAllGlobalOptionsTest) {
    GEThreadLocalContext ge_local_context;
    std::map<std::string, std::string> global_maps;
    std::string key = "abc";
    std::string value = "aaa";
    global_maps.insert(std::make_pair(key, value));
    ge_local_context.SetGlobalOption(global_maps);

    std::map<std::string, std::string> global_options_;
    global_options_ = ge_local_context.GetAllGlobalOptions();
    std::string ret_value = global_options_[key];
    EXPECT_EQ(ret_value, "aaa");
}

TEST_F(UtestGeLocalContext, GetAllSessionOptionsTest) {
    GEThreadLocalContext ge_local_context;
    std::map<std::string, std::string> session_maps;
    std::string key1 = "111";
    std::string value1 = "aaa";
    std::string key2 = "222";
    std::string value2 = "bbb";
    session_maps.insert(std::make_pair(key1, value1));
    session_maps.insert(std::make_pair(key2, value2));
    ge_local_context.SetSessionOption(session_maps);

    std::map<std::string, std::string> session_options_;
    session_options_ = ge_local_context.GetAllSessionOptions();
    std::string ret_value1 = session_options_[key1];
    EXPECT_EQ(ret_value1, "aaa");
    std::string ret_value2 = session_options_[key2];
    EXPECT_EQ(ret_value2, "bbb");
}

TEST_F(UtestGeLocalContext, GetAllGraphOptionsTest) {
    GEThreadLocalContext ge_local_context;
    std::map<std::string, std::string> graph_maps;
    std::string key1 = "333";
    std::string value1 = "cccc";
    std::string key2 = "444";
    std::string value2 = "ddd";
    graph_maps.insert(std::make_pair(key1, value1));
    graph_maps.insert(std::make_pair(key2, value2));
    ge_local_context.SetGraphOption(graph_maps);

    std::map<std::string, std::string> graph_options_;
    graph_options_ = ge_local_context.GetAllGraphOptions();
    std::string ret_value1 = graph_options_[key1];
    EXPECT_EQ(ret_value1, "cccc");
    std::string ret_value2 = graph_options_[key2];
    EXPECT_EQ(ret_value2, "ddd");
}

TEST_F(UtestGeLocalContext, GetAllOptionsTest) {
    GEThreadLocalContext ge_local_context;
    std::map<std::string, std::string> global_maps;
    std::string global_key1 = "111";
    std::string global_value1 = "aaa";
    std::string global_key2 = "222";
    std::string global_value2 = "bbb";
    global_maps.insert(std::make_pair(global_key1, global_value1));
    global_maps.insert(std::make_pair(global_key2, global_value2));
    ge_local_context.SetGlobalOption(global_maps);

    std::map<std::string, std::string> session_maps;
    std::string session_key1 = "333";
    std::string session_value1 = "ccc";
    std::string session_key2 = "444";
    std::string session_value2 = "ddd";
    session_maps.insert(std::make_pair(session_key1, session_value1));
    session_maps.insert(std::make_pair(session_key2, session_value2));
    ge_local_context.SetSessionOption(session_maps);

    std::map<std::string, std::string> graph_maps;
    std::string graph_key1 = "555";
    std::string graph_value1 = "eee";
    std::string graph_key2 = "666";
    std::string graph_value2 = "fff";
    graph_maps.insert(std::make_pair(graph_key1, graph_value1));
    graph_maps.insert(std::make_pair(graph_key2, graph_value2));
    ge_local_context.SetGraphOption(graph_maps);

    std::map<std::string, std::string> options_all;
    options_all = ge_local_context.GetAllOptions();
    std::string ret_value1 = options_all["222"];
    EXPECT_EQ(ret_value1, "bbb");
    std::string ret_value2 = options_all["444"];
    EXPECT_EQ(ret_value2, "ddd");
    std::string ret_value3 = options_all["555"];
    EXPECT_EQ(ret_value3, "eee");
}
} // namespace ge