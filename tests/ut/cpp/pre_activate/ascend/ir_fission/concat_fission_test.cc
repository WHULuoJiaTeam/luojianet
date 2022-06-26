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

#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"
#define private public
#define protected public
#include "plugin/device/ascend/optimizer/ir_fission/concat_fission.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class TestHWConcatFission : public BackendCommon {
 public:
  TestHWConcatFission() : get_py_fun_("gtest_input.pre_activate.concat_fission_test", true) {}
  ~TestHWConcatFission() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWConcatFission, test_concat_fission_divided_by_2) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_concat_fission", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 9; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto concat_fission = std::make_shared<opt::ConcatFission>();
  concat_fission->inputs_divisor_ = 2;
  pm->AddPass(concat_fission);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_concat_fission", "after_divided_by_2");
  EXPECT_NE(g_after, nullptr);
  auto kg_after = GetKernelGraph(g_after, args_spec_list);
  EXPECT_TRUE(CheckEqualGraph(kg_after, new_graph));
}

TEST_F(TestHWConcatFission, test_concat_fission_divided_by_3) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_concat_fission", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 9; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto concat_fission = std::make_shared<opt::ConcatFission>();
  concat_fission->inputs_divisor_ = 3;
  pm->AddPass(concat_fission);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_concat_fission", "after_divided_by_3");
  EXPECT_NE(g_after, nullptr);
  auto kg_after = GetKernelGraph(g_after, args_spec_list);
  EXPECT_TRUE(CheckEqualGraph(kg_after, new_graph));
}

TEST_F(TestHWConcatFission, test_concat_fission_divided_by_4) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_concat_fission", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 9; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto concat_fission = std::make_shared<opt::ConcatFission>();
  concat_fission->inputs_divisor_ = 4;
  pm->AddPass(concat_fission);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_concat_fission", "after_divided_by_4");
  EXPECT_NE(g_after, nullptr);
  auto kg_after = GetKernelGraph(g_after, args_spec_list);
  EXPECT_TRUE(CheckEqualGraph(kg_after, new_graph));
}

TEST_F(TestHWConcatFission, test_concat_fission_divided_by_8) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_concat_fission", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 9; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto concat_fission = std::make_shared<opt::ConcatFission>();
  concat_fission->inputs_divisor_ = 8;
  pm->AddPass(concat_fission);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_concat_fission", "after_divided_by_8");
  EXPECT_NE(g_after, nullptr);
  auto kg_after = GetKernelGraph(g_after, args_spec_list);
  EXPECT_TRUE(CheckEqualGraph(kg_after, new_graph));
}

TEST_F(TestHWConcatFission, test_concat_fission_divided_by_9) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_concat_fission", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 9; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto concat_fission = std::make_shared<opt::ConcatFission>();
  concat_fission->inputs_divisor_ = 9;
  pm->AddPass(concat_fission);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_concat_fission", "after_divided_by_9");
  EXPECT_NE(g_after, nullptr);
  auto kg_after = GetKernelGraph(g_after, args_spec_list);
  EXPECT_TRUE(CheckEqualGraph(kg_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
