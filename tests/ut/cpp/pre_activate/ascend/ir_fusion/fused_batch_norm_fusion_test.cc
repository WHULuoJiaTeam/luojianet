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

#include "plugin/device/ascend/optimizer/ir_fusion/fused_batch_norm_fusion.h"
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"

namespace mindspore {
namespace opt {
class TestHWFusedBatchNormFusion : public BackendCommon {
 public:
  TestHWFusedBatchNormFusion() : get_py_fun_("gtest_input.pre_activate.fused_batch_norm_fusion_test", true) {}
  ~TestHWFusedBatchNormFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWFusedBatchNormFusion, test_fused_batch_norm_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_fused_batch_norm_fusion", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{32, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{64};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract};
  for (size_t i = 0; i < 6; ++i) {
    args_spec_list.push_back(y_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::FusedBatchNormFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_fused_batch_norm_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWFusedBatchNormFusion, test_fused_batch_norm_mix_precision_fusion0) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_fused_batch_norm_fusion", "before_mix_precision0");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{32, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{64};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract};
  for (size_t i = 0; i < 6; ++i) {
    args_spec_list.push_back(y_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::FusedBatchNormMixPrecisionFusion0>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_fused_batch_norm_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWFusedBatchNormFusion, test_fused_batch_norm_mix_precision_fusion1) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_fused_batch_norm_fusion", "before_mix_precision1");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{32, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{64};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract};
  for (size_t i = 0; i < 6; ++i) {
    args_spec_list.push_back(y_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::FusedBatchNormMixPrecisionFusion1>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_fused_batch_norm_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore