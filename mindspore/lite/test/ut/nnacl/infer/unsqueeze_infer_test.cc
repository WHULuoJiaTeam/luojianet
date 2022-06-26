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
#include "common/common_test.h"
#include "nnacl/infer/unsqueeze_infer.h"
#include "nnacl/unsqueeze_parameter.h"

namespace mindspore {

class UnsqueezeInferTest : public mindspore::CommonTest {
 public:
  UnsqueezeInferTest() {}
};

TEST_F(UnsqueezeInferTest, UnsqueezeInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  UnSqueezeParameter *parameter = new UnSqueezeParameter;
  parameter->num_dim_ = 1;
  parameter->dims_[0] = 0;
  int ret = UnsqueezeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(UnsqueezeInferTest, UnsqueezeInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  UnSqueezeParameter *parameter = new UnSqueezeParameter;
  parameter->num_dim_ = 1;
  parameter->dims_[0] = 1;
  int ret = UnsqueezeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(UnsqueezeInferTest, UnsqueezeInferTest2) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  UnSqueezeParameter *parameter = new UnSqueezeParameter;
  parameter->num_dim_ = 0;
  int ret = UnsqueezeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(UnsqueezeInferTest, UnsqueezeInferTest3) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 6;
  inputs[0]->shape_[3] = 7;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  UnSqueezeParameter *parameter = new UnSqueezeParameter;
  parameter->num_dim_ = 1;
  parameter->dims_[0] = 1;
  int ret = UnsqueezeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 5);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
  ASSERT_EQ(outputs[0]->shape_[2], 5);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  ASSERT_EQ(outputs[0]->shape_[4], 7);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(UnsqueezeInferTest, UnsqueezeInferTest4) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 6;
  inputs[0]->shape_[3] = 7;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  UnSqueezeParameter *parameter = new UnSqueezeParameter;
  parameter->num_dim_ = 1;
  parameter->dims_[0] = 1;
  int ret = UnsqueezeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 5);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
  ASSERT_EQ(outputs[0]->shape_[2], 5);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  ASSERT_EQ(outputs[0]->shape_[4], 7);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(UnsqueezeInferTest, UnsqueezeInferTest5) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 6;
  inputs[0]->shape_[3] = 7;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  UnSqueezeParameter *parameter = new UnSqueezeParameter;
  parameter->num_dim_ = 1;
  parameter->dims_[0] = 3;
  int ret = UnsqueezeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 5);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
  ASSERT_EQ(outputs[0]->shape_[2], 6);
  ASSERT_EQ(outputs[0]->shape_[3], 1);
  ASSERT_EQ(outputs[0]->shape_[4], 7);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
