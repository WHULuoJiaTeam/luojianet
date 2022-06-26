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
#include <memory>
#include "common/common_test.h"
#include "include/api/context.h"

namespace mindspore {
class TestCxxApiContext : public UT::Common {
 public:
  TestCxxApiContext() = default;
};

TEST_F(TestCxxApiContext, test_context_device_info_cast_SUCCESS) {
  std::shared_ptr<DeviceInfoContext> cpu = std::make_shared<CPUDeviceInfo>();
  std::shared_ptr<DeviceInfoContext> gpu = std::make_shared<GPUDeviceInfo>();
  std::shared_ptr<DeviceInfoContext> kirin_npu = std::make_shared<KirinNPUDeviceInfo>();
  std::shared_ptr<DeviceInfoContext> ascend = std::make_shared<AscendDeviceInfo>();

  ASSERT_TRUE(cpu->Cast<CPUDeviceInfo>() != nullptr);
  ASSERT_TRUE(gpu->Cast<GPUDeviceInfo>() != nullptr);
  ASSERT_TRUE(kirin_npu->Cast<KirinNPUDeviceInfo>() != nullptr);
  ASSERT_TRUE(ascend->Cast<AscendDeviceInfo>() != nullptr);
}

TEST_F(TestCxxApiContext, test_context_device_info_cast_FAILED) {
  std::shared_ptr<DeviceInfoContext> cpu = std::make_shared<CPUDeviceInfo>();
  std::shared_ptr<DeviceInfoContext> gpu = std::make_shared<GPUDeviceInfo>();
  std::shared_ptr<DeviceInfoContext> kirin_npu = std::make_shared<KirinNPUDeviceInfo>();
  std::shared_ptr<DeviceInfoContext> ascend = std::make_shared<AscendDeviceInfo>();

  ASSERT_TRUE(cpu->Cast<GPUDeviceInfo>() == nullptr);
  ASSERT_TRUE(kirin_npu->Cast<GPUDeviceInfo>() == nullptr);
  ASSERT_TRUE(ascend->Cast<GPUDeviceInfo>() == nullptr);

  ASSERT_TRUE(gpu->Cast<CPUDeviceInfo>() == nullptr);
  ASSERT_TRUE(kirin_npu->Cast<CPUDeviceInfo>() == nullptr);
  ASSERT_TRUE(ascend->Cast<CPUDeviceInfo>() == nullptr);
}

TEST_F(TestCxxApiContext, test_context_get_set_SUCCESS) {
  int32_t thread_num = 22;
  auto context = std::make_shared<Context>();
  context->SetThreadNum(thread_num);
  ASSERT_EQ(context->GetThreadNum(), thread_num);
}

TEST_F(TestCxxApiContext, test_context_cpu_context_SUCCESS) {
  auto context = std::make_shared<Context>();
  std::shared_ptr<CPUDeviceInfo> cpu = std::make_shared<CPUDeviceInfo>();
  cpu->SetEnableFP16(true);
  context->MutableDeviceInfo().push_back(cpu);
  ASSERT_EQ(context->MutableDeviceInfo().size(), 1);
  auto cpu_2 = context->MutableDeviceInfo()[0]->Cast<CPUDeviceInfo>();
  ASSERT_TRUE(cpu_2 != nullptr);
  ASSERT_TRUE(cpu_2->GetEnableFP16());
}

TEST_F(TestCxxApiContext, test_context_ascend_context_FAILED) {
  std::string option_1 = "aaa";
  std::string option_2 = "vvv";
  std::string option_3 = "www";
  std::string option_4 = "rrr";
  std::string option_5 = "ppp";
  std::string option_6 = "sss";
  uint32_t option_7 = 77;
  enum DataType option_8 = DataType::kNumberTypeInt16;
  std::vector<size_t> option_9 = {1, 2, 3, 4, 5};
  std::string option_9_ans = "1,2,3,4,5";

  auto context = std::make_shared<Context>();
  std::shared_ptr<AscendDeviceInfo> ascend310 = std::make_shared<AscendDeviceInfo>();
  ascend310->SetInputShape(option_1);
  ascend310->SetInsertOpConfigPath(option_2);
  ascend310->SetOpSelectImplMode(option_3);
  ascend310->SetPrecisionMode(option_4);
  ascend310->SetInputFormat(option_5);
  ascend310->SetFusionSwitchConfigPath(option_6);
  ascend310->SetDeviceID(option_7);
  ascend310->SetOutputType(option_8);
  ascend310->SetDynamicBatchSize(option_9);

  context->MutableDeviceInfo().push_back(ascend310);
  ASSERT_EQ(context->MutableDeviceInfo().size(), 1);
  auto ctx = context->MutableDeviceInfo()[0]->Cast<AscendDeviceInfo>();
  ASSERT_TRUE(ctx != nullptr);
  ASSERT_EQ(ascend310->GetInputShape(), option_1);
  ASSERT_EQ(ascend310->GetInsertOpConfigPath(), option_2);
  ASSERT_EQ(ascend310->GetOpSelectImplMode(), option_3);
  ASSERT_EQ(ascend310->GetPrecisionMode(), option_4);
  ASSERT_EQ(ascend310->GetInputFormat(), option_5);
  ASSERT_EQ(ascend310->GetFusionSwitchConfigPath(), option_6);
  ASSERT_EQ(ascend310->GetDeviceID(), option_7);
  ASSERT_EQ(ascend310->GetOutputType(), option_8);
  ASSERT_EQ(ascend310->GetDynamicBatchSize(), option_9_ans);
}

TEST_F(TestCxxApiContext, test_context_ascend310_context_default_value_SUCCESS) {
  auto ctx = std::make_shared<AscendDeviceInfo>();
  ASSERT_EQ(ctx->GetOpSelectImplMode(), "");
}
}  // namespace mindspore
