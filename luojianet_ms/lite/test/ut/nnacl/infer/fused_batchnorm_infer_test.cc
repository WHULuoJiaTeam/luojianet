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
#include "common/common_test.h"
#include "nnacl/infer/fused_batchnorm_infer.h"

namespace luojianet_ms {

class FusedBatchNormInferTest : public luojianet_ms::CommonTest {
 public:
  FusedBatchNormInferTest() {}
};

TEST_F(FusedBatchNormInferTest, FusedBatchNormInferTest0) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 5;
  inputs[2] = new TensorC;
  inputs[2]->shape_size_ = 2;
  inputs[2]->shape_[0] = 8;
  inputs[2]->shape_[1] = 7;
  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  OpParameter *parameter = new OpParameter;
  int ret = FusedBatchNormInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                     reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[1]->shape_size_, 2);
  ASSERT_EQ(outputs[1]->shape_[0], 6);
  ASSERT_EQ(outputs[1]->shape_[1], 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace luojianet_ms
