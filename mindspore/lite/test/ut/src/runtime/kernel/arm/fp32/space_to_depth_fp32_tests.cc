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
#include <vector>
#include <iostream>
#include <memory>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "nnacl/space_to_depth_parameter.h"
#include "nnacl/base/space_to_depth_base.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {

class SpaceToDepthTestFp32 : public mindspore::CommonTest {
 public:
  SpaceToDepthTestFp32() {}
};

TEST_F(SpaceToDepthTestFp32, SpaceToDepthTest1) {
  float input[16] = {1, 2, 5, 6, 10, 20, 3, 8, 18, 10, 3, 4, 11, 55, 15, 25};
  const int out_size = 16;
  float expect_out[16] = {1, 2, 10, 20, 5, 6, 3, 8, 18, 10, 11, 55, 3, 4, 15, 25};

  float output[16];
  int in_shape[4] = {1, 4, 4, 1};
  int out_shape[4] = {1, 2, 2, 4};
  SpaceToDepthParameter param;
  param.op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  param.op_parameter_.thread_num_ = 1;
  param.block_size_ = 2;
  param.date_type_len = sizeof(float);

  SpaceToDepthForNHWC(input, output, in_shape, out_shape, 4, &param, 0);
  for (int i = 0; i < out_size; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, out_size, 0.000001));
}

TEST_F(SpaceToDepthTestFp32, SpaceToDepthTest2) {
  std::vector<float> input = {1, 2, 5, 6, 10, 20, 3, 8, 18, 10, 3, 4, 11, 55, 15, 25};
  std::vector<int> in_shape = {1, 4, 4, 1};
  lite::Tensor input_tensor;
  input_tensor.set_data(input.data());
  input_tensor.set_shape(in_shape);
  input_tensor.set_format(mindspore::NHWC);
  input_tensor.set_data_type(kNumberTypeFloat32);
  std::vector<lite::Tensor *> inputs_tensor;
  inputs_tensor.push_back(&input_tensor);

  const int out_size = 16;
  float expect_out[16] = {1, 2, 10, 20, 5, 6, 3, 8, 18, 10, 11, 55, 3, 4, 15, 25};
  std::vector<float> output(16);
  std::vector<int> out_shape = {1, 2, 2, 4};
  lite::Tensor output_tensor;
  output_tensor.set_data(output.data());
  output_tensor.set_shape(out_shape);
  output_tensor.set_format(mindspore::NHWC);
  output_tensor.set_data_type(kNumberTypeFloat32);
  std::vector<lite::Tensor *> outputs_tensor;
  outputs_tensor.push_back(&output_tensor);

  auto param = static_cast<SpaceToDepthParameter *>(malloc(sizeof(SpaceToDepthParameter)));
  param->op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  param->op_parameter_.thread_num_ = 1;
  param->block_size_ = 2;
  param->date_type_len = sizeof(float);

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_SpaceToDepth};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(param), ctx.get(), desc);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  for (int i = 0; i < out_size; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output.data(), expect_out, out_size, 0.000001));
  input_tensor.set_data(nullptr);
  output_tensor.set_data(nullptr);
  delete kernel;
}

}  // namespace mindspore
