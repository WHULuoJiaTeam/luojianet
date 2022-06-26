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
#include "common/common.h"
#include "minddata/dataset/kernels/data/pad_end_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestPadEndOp : public UT::Common {
 protected:
  MindDataTestPadEndOp() {}
};

TEST_F(MindDataTestPadEndOp, TestOp) {
  MS_LOG(INFO) << "Doing MindDataTestPadEndOp.";

  // first set of testunits for numeric values

  TensorShape pad_data_shape({});

  // prepare input tensor
  std::vector<float> orig1 = {1, 1, 1, 1};
  TensorShape input_shape1({2, 2});
  std::vector<TensorShape> input_shape1_vector = {input_shape1};
  std::shared_ptr<Tensor> input1;
  Tensor::CreateFromVector(orig1, input_shape1, &input1);

  // pad_shape
  TensorShape pad_shape1[3] = {TensorShape({3, 3}), TensorShape({2, 4}), TensorShape({4, 2})};

  // value to pad
  std::vector<std::vector<float>> pad_data1 = {{0}, {3.5}, {3.5}};

  std::shared_ptr<Tensor> expected1[3];

  // expected tensor output for testunit 1
  std::vector<float> out1 = {1, 1, 0, 1, 1, 0, 0, 0, 0};
  Tensor::CreateFromVector(out1, pad_shape1[0], &(expected1[0]));

  // expected tensor output for testunit 2
  std::vector<float> out2 = {1, 1, 3.5, 3.5, 1, 1, 3.5, 3.5};
  Tensor::CreateFromVector(out2, pad_shape1[1], &(expected1[1]));

  // expected tensor output for testunit 3
  std::vector<float> out3 = {1, 1, 1, 1, 3.5, 3.5, 3.5, 3.5};
  Tensor::CreateFromVector(out3, pad_shape1[2], &(expected1[2]));

  // run the PadEndOp
  for (auto i = 0; i < 3; i++) {
    std::shared_ptr<Tensor> output;
    std::vector<TensorShape> output_shape = {TensorShape({})};

    std::shared_ptr<Tensor> pad_value1;
    Tensor::CreateFromVector(pad_data1[i], pad_data_shape, &pad_value1);

    std::unique_ptr<PadEndOp> op(new PadEndOp(pad_shape1[i], pad_value1));
    Status s = op->Compute(input1, &output);

    EXPECT_TRUE(s.IsOk());
    ASSERT_TRUE(output->shape() == expected1[i]->shape());
    ASSERT_TRUE(output->type() == expected1[i]->type());
    MS_LOG(DEBUG) << *output << std::endl;
    MS_LOG(DEBUG) << *expected1[i] << std::endl;
    ASSERT_TRUE(*output == *expected1[i]);

    s = op->OutputShape(input_shape1_vector, output_shape);
    EXPECT_TRUE(s.IsOk());
    ASSERT_TRUE(output_shape.size() == 1);
    ASSERT_TRUE(output->shape() == output_shape[0]);
  }

  // second set of testunits for string

  // input tensor
  std::vector<std::string> orig2 = {"this", "is"};
  TensorShape input_shape2({2});
  std::vector<TensorShape> input_shape2_vector = {input_shape2};
  std::shared_ptr<Tensor> input2;
  Tensor::CreateFromVector(orig2, input_shape2, &input2);

  // pad_shape
  TensorShape pad_shape2[3] = {TensorShape({5}), TensorShape({2}), TensorShape({10})};

  // pad value
  std::vector<std::string> pad_data2[3] = {{""}, {"P"}, {" "}};
  std::shared_ptr<Tensor> pad_value2[3];

  // expected output for 3 testunits
  std::shared_ptr<Tensor> expected2[3];
  std::vector<std::string> outstring[3] = {
    {"this", "is", "", "", ""}, {"this", "is"}, {"this", "is", " ", " ", " ", " ", " ", " ", " ", " "}};

  for (auto i = 0; i < 3; i++) {
    // pad value
    Tensor::CreateFromVector(pad_data2[i], pad_data_shape, &pad_value2[i]);

    std::shared_ptr<Tensor> output;
    std::vector<TensorShape> output_shape = {TensorShape({})};

    std::unique_ptr<PadEndOp> op(new PadEndOp(pad_shape2[i], pad_value2[i]));

    Status s = op->Compute(input2, &output);

    Tensor::CreateFromVector(outstring[i], pad_shape2[i], &expected2[i]);

    EXPECT_TRUE(s.IsOk());
    ASSERT_TRUE(output->shape() == expected2[i]->shape());
    ASSERT_TRUE(output->type() == expected2[i]->type());
    MS_LOG(DEBUG) << *output << std::endl;
    MS_LOG(DEBUG) << *expected2[i] << std::endl;
    ASSERT_TRUE(*output == *expected2[i]);

    s = op->OutputShape(input_shape2_vector, output_shape);
    EXPECT_TRUE(s.IsOk());
    ASSERT_TRUE(output_shape.size() == 1);
    ASSERT_TRUE(output->shape() == output_shape[0]);
  }

  MS_LOG(INFO) << "MindDataTestPadEndOp end.";
}
