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
#include "backend/common/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ir_fusion/confusion_mul_grad_fusion.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
class TestHWOptimizeConfusionMulGradFusion : public BackendCommon {
 public:
  TestHWOptimizeConfusionMulGradFusion() : get_py_fun_("gtest_input.pre_activate.confusion_mul_grad_fusion", true) {}
  ~TestHWOptimizeConfusionMulGradFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWOptimizeConfusionMulGradFusion, test_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_confusion_mul_grad_fusion", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp{10, 1024};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 3; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConfusionMulGradFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_confusion_mul_grad_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
