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
#include "nnacl/infer/strided_slice_infer.h"

namespace mindspore {

class StridedSliceInferTest : public mindspore::CommonTest {
 public:
  StridedSliceInferTest() {}
};

TEST_F(StridedSliceInferTest, StridedSliceInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->begins_[0] = 1;
  parameter->begins_[1] = 0;
  parameter->begins_[2] = 0;
  parameter->ends_[0] = 2;
  parameter->ends_[1] = 1;
  parameter->ends_[2] = 3;
  parameter->strides_[0] = 1;
  parameter->strides_[1] = 1;
  parameter->strides_[2] = 1;
  parameter->num_axes_ = 3;
  parameter->begins_mask_ = 0;
  parameter->ends_mask_ = 0;
  parameter->ellipsisMask_ = 0;
  parameter->newAxisMask_ = 0;
  parameter->shrinkAxisMask_ = 0;
  int ret = StridedSliceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(StridedSliceInferTest, StridedSliceInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->begins_[0] = 1;
  parameter->begins_[1] = 0;
  parameter->begins_[2] = 0;
  parameter->ends_[0] = 2;
  parameter->ends_[1] = 2;
  parameter->ends_[2] = 3;
  parameter->strides_[0] = 1;
  parameter->strides_[1] = 1;
  parameter->strides_[2] = 1;
  parameter->num_axes_ = 3;
  parameter->begins_mask_ = 0;
  parameter->ends_mask_ = 0;
  parameter->ellipsisMask_ = 0;
  parameter->newAxisMask_ = 0;
  parameter->shrinkAxisMask_ = 0;
  int ret = StridedSliceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(StridedSliceInferTest, StridedSliceInferTest2) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->begins_[0] = 1;
  parameter->begins_[1] = -1;
  parameter->begins_[2] = 0;
  parameter->ends_[0] = 2;
  parameter->ends_[1] = -3;
  parameter->ends_[2] = 3;
  parameter->strides_[0] = 1;
  parameter->strides_[1] = -1;
  parameter->strides_[2] = 1;
  parameter->num_axes_ = 3;
  parameter->begins_mask_ = 0;
  parameter->ends_mask_ = 0;
  parameter->ellipsisMask_ = 0;
  parameter->newAxisMask_ = 0;
  parameter->shrinkAxisMask_ = 0;
  int ret = StridedSliceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(StridedSliceInferTest, StridedSliceInferTest3) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 5;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->begins_[0] = 0;
  parameter->ends_[0] = 3;
  parameter->strides_[0] = 1;
  parameter->num_axes_ = 1;
  parameter->begins_mask_ = 0;
  parameter->ends_mask_ = 0;
  parameter->ellipsisMask_ = 0;
  parameter->newAxisMask_ = 0;
  parameter->shrinkAxisMask_ = 0;
  int ret = StridedSliceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(StridedSliceInferTest, StridedSliceInferTest4) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 5;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->begins_[0] = 1;
  parameter->ends_[0] = -2;
  parameter->strides_[0] = 1;
  parameter->num_axes_ = 1;
  parameter->begins_mask_ = 0;
  parameter->ends_mask_ = 0;
  parameter->ellipsisMask_ = 0;
  parameter->newAxisMask_ = 0;
  parameter->shrinkAxisMask_ = 0;
  int ret = StridedSliceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 2);

  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(StridedSliceInferTest, StridedSliceInferTest5) {
  size_t inputs_size = 4;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 5;
  // std::vector<int> begin_vector = {1};
  // std::vector<int> end_vector = {-2};
  // std::vector<int> stride_vector = {1};
  int *begin_vector = reinterpret_cast<int *>(malloc(sizeof(int)));
  begin_vector[0] = 1;
  int *end_vector = reinterpret_cast<int *>(malloc(sizeof(int)));
  end_vector[0] = -2;
  int *stride_vector = reinterpret_cast<int *>(malloc(sizeof(int)));
  stride_vector[0] = 1;
  inputs[1] = new TensorC;
  // inputs[1]->data_ = begin_vector.data();
  inputs[1]->data_ = begin_vector;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 1;
  inputs[2] = new TensorC;
  inputs[2]->data_ = end_vector;
  inputs[2]->shape_size_ = 1;
  inputs[2]->shape_[0] = 1;
  inputs[3] = new TensorC;
  inputs[3]->data_ = stride_vector;
  inputs[3]->shape_size_ = 1;
  inputs[3]->shape_[0] = 1;
  std::vector<TensorC *> outputs;
  outputs.push_back(NULL);
  outputs[0] = new TensorC;
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->begins_mask_ = 0;
  parameter->ends_mask_ = 0;
  parameter->ellipsisMask_ = 0;
  parameter->newAxisMask_ = 0;
  parameter->shrinkAxisMask_ = 0;
  int ret = StridedSliceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  delete parameter;
  delete inputs[0];
  delete inputs[1];
  delete inputs[2];
  delete inputs[3];
  delete outputs[0];
  free(begin_vector);
  free(end_vector);
  free(stride_vector);
}

TEST_F(StridedSliceInferTest, StridedSliceInferTest6) {
  size_t inputs_size = 4;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 3;
  std::vector<int> begin_vector = {1, 0, 0};
  std::vector<int> end_vector = {2, 1, 3};
  std::vector<int> stride_vector = {1, 1, 1};
  inputs[1] = new TensorC;
  inputs[1]->data_ = begin_vector.data();
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 3;
  inputs[2] = new TensorC;
  inputs[2]->data_ = end_vector.data();
  inputs[2]->shape_size_ = 1;
  inputs[2]->shape_[0] = 3;
  inputs[3] = new TensorC;
  inputs[3]->data_ = stride_vector.data();
  inputs[3]->shape_size_ = 1;
  inputs[3]->shape_[0] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  StridedSliceParameter *parameter = new StridedSliceParameter;
  parameter->begins_mask_ = 0;
  parameter->ends_mask_ = 0;
  parameter->ellipsisMask_ = 0;
  parameter->newAxisMask_ = 0;
  parameter->shrinkAxisMask_ = 0;
  int ret = StridedSliceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  delete parameter;
}

}  // namespace mindspore
