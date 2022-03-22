/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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
#include "graph/operator_factory_impl.h"
#include "init/gelib.h"
#include "external/ge/ge_api.h"
#include "opskernel_manager/ops_kernel_builder_manager.h"
#include "ge_running_env/fake_ops_kernel_builder.h"
#include "ge_running_env/fake_ns.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
FAKE_NS_BEGIN

#define ASSERT_OPS_LIST_SIZE(list_size) \
  std::vector<AscendString> ops_list; \
  OperatorFactory::GetOpsTypeList(ops_list);\
  ASSERT_EQ(ops_list.size(), list_size);

class GeRunningEvnFakerTest : public testing::Test {
 protected:
  void SetUp() {}
  OpsKernelManager &kernel_manager = ge::GELib::GetInstance()->OpsKernelManagerObj();
  OpsKernelBuilderManager &builder_manager = OpsKernelBuilderManager::Instance();
};

TEST_F(GeRunningEvnFakerTest, test_reset_running_env_is_success) {
  GeRunningEnvFaker ge_env;
  ge_env.Reset();
  ASSERT_OPS_LIST_SIZE(0);
  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfoStores().size(), 1);
  ASSERT_EQ(builder_manager.GetAllOpsKernelBuilders().size(), 1);
  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfo().size(), 52);
  ASSERT_EQ(kernel_manager.GetOpsKernelInfo(SWITCH).size(), 1);
}

TEST_F(GeRunningEvnFakerTest, test_install_fake_op_success) {
  GeRunningEnvFaker ge_env;
  ge_env.Install(FakeOp(DATA)).Install(FakeOp(SWITCH));
  ASSERT_OPS_LIST_SIZE(2);
  ASSERT_TRUE(OperatorFactory::IsExistOp(DATA));
  ASSERT_TRUE(OperatorFactory::IsExistOp(SWITCH));
}

TEST_F(GeRunningEvnFakerTest, test_install_fake_op_with_inputs_and_outputs_success) {
  GeRunningEnvFaker ge_env;
  ge_env.Install(FakeOp(ADD).Inputs({"x1", "x2"}).Outputs({"y"}));

  auto add1 = OperatorFactory::CreateOperator("add1", ADD);

  ASSERT_EQ(add1.GetInputsSize(), 2);
  ASSERT_EQ(add1.GetOutputsSize(), 1);
  ASSERT_OPS_LIST_SIZE(1);
}

TEST_F(GeRunningEvnFakerTest, test_install_fake_op_with_infer_shape_success) {
  GeRunningEnvFaker ge_env;
  auto infer_fun = [](Operator &op) -> graphStatus {
    TensorDesc input_desc = op.GetInputDescByName("data");
    return GRAPH_SUCCESS;
  };
  ASSERT_TRUE(OperatorFactoryImpl::GetInferShapeFunc(DATA) == nullptr);

  ge_env.Install(FakeOp(DATA).Inputs({"data"}).InferShape(infer_fun));

  ASSERT_TRUE(OperatorFactoryImpl::GetInferShapeFunc(DATA) != nullptr);
}

TEST_F(GeRunningEvnFakerTest, test_install_engine_with_default_info_store) {
  GeRunningEnvFaker ge_env;
  ge_env.Install(FakeEngine("DNN_HCCL"));

  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfoStores().size(), 2);
  ASSERT_EQ(builder_manager.GetAllOpsKernelBuilders().size(), 2);
  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfo().size(), 52);
  ASSERT_EQ(kernel_manager.GetOpsKernelInfo(SWITCH).size(), 1);
}

TEST_F(GeRunningEvnFakerTest, test_install_engine_with_info_store_name) {
  GeRunningEnvFaker ge_env;
  ge_env.Install(FakeEngine("DNN_HCCL").KernelInfoStore("AiCoreLib2"))
    .Install(FakeOp(SWITCH).InfoStoreAndBuilder("AiCoreLib2"));

  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfoStores().size(), 2);
  ASSERT_EQ(builder_manager.GetAllOpsKernelBuilders().size(), 2);
  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfo().size(), 52);
  ASSERT_EQ(kernel_manager.GetOpsKernelInfo(SWITCH).size(), 2);
}

TEST_F(GeRunningEvnFakerTest, test_install_custom_kernel_builder_success) {
  struct FakeKernelBuilder : FakeOpsKernelBuilder {
    Status CalcOpRunningParam(Node &node) override {
      OpDescPtr op_desc = node.GetOpDesc();
      if (op_desc == nullptr) {
        return FAILED;
      }
      return SUCCESS;
    }
  };

  GeRunningEnvFaker ge_env;
  auto ai_core_kernel = FakeEngine("DNN_HCCL").KernelBuilder(std::make_shared<FakeKernelBuilder>());
  ge_env.Reset().Install(ai_core_kernel);

  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfoStores().size(), 2);
  ASSERT_EQ(builder_manager.GetAllOpsKernelBuilders().size(), 2);
  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfo().size(), 52);
}

TEST_F(GeRunningEvnFakerTest, test_install_custom_kernel_info_store_success) {
  struct FakeKernelBuilder : FakeOpsKernelInfoStore {
    FakeKernelBuilder(const std::string &kernel_lib_name) : FakeOpsKernelInfoStore(kernel_lib_name) {}

    bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override { return FAILED; }
  };

  GeRunningEnvFaker ge_env;
  auto ai_core_kernel = FakeEngine("DNN_HCCL").KernelInfoStore(std::make_shared<FakeKernelBuilder>("AiCoreLib2"));
  ge_env.Reset().Install(ai_core_kernel);

  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfoStores().size(), 2);
  ASSERT_EQ(builder_manager.GetAllOpsKernelBuilders().size(), 2);
  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfo().size(), 52);
}

TEST_F(GeRunningEvnFakerTest, test_install_default_fake_engine_success) {
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();

  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfoStores().size(), 7);
  ASSERT_EQ(builder_manager.GetAllOpsKernelBuilders().size(), 7);
  ASSERT_EQ(kernel_manager.GetAllOpsKernelInfo().size(), 66);
}

FAKE_NS_END
