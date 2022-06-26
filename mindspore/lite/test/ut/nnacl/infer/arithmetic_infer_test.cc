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
#include "nnacl/infer/arithmetic_infer.h"

namespace mindspore {

class ArithmeticInferTest : public mindspore::CommonTest {
 public:
  ArithmeticInferTest() {}
};

TEST_F(ArithmeticInferTest, ArithmeticInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 4;
  inputs[0]->shape_[3] = 5;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 5;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 7;
  inputs[1]->shape_[2] = 8;
  inputs[1]->shape_[3] = 9;
  inputs[1]->shape_[4] = 10;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  int ret =
    ArithmeticInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(), parameter);
  ASSERT_EQ(ret, NNACL_ERR);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ArithmeticInferTest, ArithmeticInferTest1) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 7;
  inputs[0]->shape_[1] = 8;
  inputs[0]->shape_[2] = 9;
  inputs[0]->shape_[3] = 10;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 5;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 7;
  inputs[1]->shape_[2] = 8;
  inputs[1]->shape_[3] = 9;
  inputs[1]->shape_[4] = 10;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  int ret =
    ArithmeticInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(), parameter);
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 5);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  ASSERT_EQ(outputs[0]->shape_[1], 7);
  ASSERT_EQ(outputs[0]->shape_[2], 8);
  ASSERT_EQ(outputs[0]->shape_[3], 9);
  ASSERT_EQ(outputs[0]->shape_[4], 10);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ArithmeticInferTest, ArithmeticInferTest2) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 7;
  inputs[1]->shape_[1] = 8;
  inputs[1]->shape_[2] = 9;
  inputs[1]->shape_[3] = 10;
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 5;
  inputs[0]->shape_[0] = 6;
  inputs[0]->shape_[1] = 7;
  inputs[0]->shape_[2] = 8;
  inputs[0]->shape_[3] = 9;
  inputs[0]->shape_[4] = 10;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  int ret =
    ArithmeticInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(), parameter);
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 5);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  ASSERT_EQ(outputs[0]->shape_[1], 7);
  ASSERT_EQ(outputs[0]->shape_[2], 8);
  ASSERT_EQ(outputs[0]->shape_[3], 9);
  ASSERT_EQ(outputs[0]->shape_[4], 10);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ArithmeticInferTest, ArithmeticInferTest3) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 5;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 7;
  inputs[1]->shape_[2] = 8;
  inputs[1]->shape_[3] = 9;
  inputs[1]->shape_[4] = 10;
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 5;
  inputs[0]->shape_[0] = 6;
  inputs[0]->shape_[1] = 7;
  inputs[0]->shape_[2] = 8;
  inputs[0]->shape_[3] = 9;
  inputs[0]->shape_[4] = 10;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  int ret =
    ArithmeticInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(), parameter);
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 5);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  ASSERT_EQ(outputs[0]->shape_[1], 7);
  ASSERT_EQ(outputs[0]->shape_[2], 8);
  ASSERT_EQ(outputs[0]->shape_[3], 9);
  ASSERT_EQ(outputs[0]->shape_[4], 10);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
