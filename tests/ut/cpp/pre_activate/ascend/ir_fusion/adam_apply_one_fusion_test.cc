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
#include "plugin/device/ascend/optimizer/ir_fusion/adam_apply_one_fusion.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
class TestHWAdamApplyOneFusion : public BackendCommon {
 public:
  TestHWAdamApplyOneFusion() : get_py_fun_("gtest_input.pre_activate.adam_apply_one_fusion_test", true) {}
  ~TestHWAdamApplyOneFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWAdamApplyOneFusion, test_adam_apply_one_fusion) {
  /*
   * def before(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, add2_y):
   *    square0 = Square(input0)
   *    mul1 = Mul(mul1_x, input0)
   *    mul0 = Mul(mul0_x, input2)
   *    mul2 = Mul(mul2_x, input1)
   *    mul3 = Mul(mul3_x, square0)
   *    add0 = Add(mul0, mul1)
   *    add1 = Add(mul2, mul3)
   *    sqrt0 = Sqrt(add1)
   *    add2 = Add(sqrt0, add2_y)
   *    true_div0 = RealDiv(add0, add2)
   *    mul4 = Mul(input4, true_div0)
   *    sub0 = Sub(input3, mul4)
   *    outputs = make_tuple(add1, add0, sub0)
   *    output = tuple_getitem(outputs, 0)
   *    return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_fusion", "before");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 10; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWAdamApplyOneFusion, test_adam_apply_one_cond1_fusion) {
  /*
   * def before_cond1(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, add2_y):
   *    square0 = Square(input0)
   *    mul1 = Mul(mul1_x, input0)
   *    mul0 = Mul(mul0_x, input2)
   *    mul2 = Mul(mul2_x, input1)
   *    mul3 = Mul(mul3_x, square0)
   *    add0 = Add(mul0, mul1)
   *    add1 = Add(mul2, mul3)
   *    sqrt0 = Sqrt(add1)
   *    add2 = Add(add2_y, sqrt0)
   *    true_div0 = RealDiv(add0, add2)
   *    mul4 = Mul(input4, true_div0)
   *    sub0 = Sub(input3, mul4)
   *    outputs = make_tuple(add1, add0, sub0)
   *    output = tuple_getitem(outputs, 0)
   *    return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_fusion", "before_cond1");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 10; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneCond1Fusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWAdamApplyOneFusion, test_adam_apply_one_cond2_fusion) {
  /*
   * def before_cond2(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, add2_y):
   *    square0 = Square(input0)
   *    mul1 = Mul(mul1_x, input0)
   *    mul0 = Mul(mul0_x, input2)
   *    mul2 = Mul(mul2_x, input1)
   *    mul3 = Mul(square0, mul3_x)
   *    add0 = Add(mul0, mul1)
   *    add1 = Add(mul2, mul3)
   *    sqrt0 = Sqrt(add1)
   *    add2 = Add(sqrt0, add2_y)
   *    true_div0 = RealDiv(add0, add2)
   *    mul4 = Mul(true_div0, input4)
   *    sub0 = Sub(input3, mul4)
   *    outputs = make_tuple(add1, add0, sub0)
   *    output = tuple_getitem(outputs, 0)
   *    return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_fusion", "before_cond2");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 10; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneCond2Fusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWAdamApplyOneFusion, test_adam_apply_one_cond3_fusion) {
  /*
   * def before_cond3(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, add2_y):
   *    square0 = Square(input0)
   *    mul1 = Mul(mul1_x, input0)
   *    mul0 = Mul(mul0_x, input2)
   *    mul2 = Mul(mul2_x, input1)
   *    mul3 = Mul(mul3_x, square0)
   *    add0 = Add(mul0, mul1)
   *    add1 = Add(mul2, mul3)
   *    sqrt0 = Sqrt(add1)
   *    add2 = Add(sqrt0, add2_y)
   *    true_div0 = RealDiv(add0, add2)
   *    mul4 = Mul(true_div0, input4)
   *    sub0 = Sub(input3, mul4)
   *    outputs = make_tuple(add1, add0, sub0)
   *    output = tuple_getitem(outputs, 0)
   *    return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_fusion", "before_cond3");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 10; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneCond3Fusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWAdamApplyOneFusion, test_adam_apply_one_cond4_fusion) {
  /*
   * def before_cond4(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, add2_y):
   *    square0 = Square(input0)
   *    mul1 = Mul(mul1_x, input0)
   *    mul0 = Mul(mul0_x, input2)
   *    mul2 = Mul(mul2_x, input1)
   *    mul3 = Mul(mul3_x, square0)
   *    add0 = Add(mul0, mul1)
   *    add1 = Add(mul2, mul3)
   *    sqrt0 = Sqrt(add1)
   *    add2 = Add(add2_y, sqrt0)
   *    true_div0 = RealDiv(add0, add2)
   *    mul4 = Mul(true_div0, input4)
   *    sub0 = Sub(input3, mul4)
   *    outputs = make_tuple(add1, add0, sub0)
   *    output = tuple_getitem(outputs, 0)
   *    return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_fusion", "before_cond4");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 10; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneCond4Fusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWAdamApplyOneFusion, test_adam_apply_one_assign_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_assign_fusion", "before");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 10; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneAssignFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_assign_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWAdamApplyOneFusion, test_adam_apply_one_assign_cond1_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_assign_fusion", "before_cond1");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 10; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneAssignCond1Fusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_assign_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWAdamApplyOneFusion, test_adam_apply_one_assign_cond2_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_assign_fusion", "before_cond2");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 10; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneAssignCond2Fusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_assign_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWAdamApplyOneFusion, test_adam_apply_one_assign_cond3_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_assign_fusion", "before_cond3");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 10; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneAssignCond3Fusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_assign_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWAdamApplyOneFusion, test_adam_apply_one_assign_cond4_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_assign_fusion", "before_cond4");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 10; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneAssignCond4Fusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_assign_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
