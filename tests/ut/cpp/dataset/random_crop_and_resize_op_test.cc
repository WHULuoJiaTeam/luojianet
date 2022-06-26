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
#include "common/common.h"
#include "common/cvop_common.h"
#include <random>
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestRandomCropAndResizeOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRandomCropAndResizeOp() : CVOpCommon() {}
};
TEST_F(MindDataTestRandomCropAndResizeOp, TestOpSimpleTest1) {
  MS_LOG(INFO) << " starting RandomCropAndResizeOp simple test";
  TensorShape s_in = input_tensor_->shape();
  TensorRow input_tensor_row;
  input_tensor_row.push_back(input_tensor_);
  input_tensor_row.push_back(input_tensor_);
  TensorRow output_tensor_row;
  std::shared_ptr<Tensor> output_tensor;
  int h_out = 1024;
  int w_out = 2048;
  float aspect_lb = 2;
  float aspect_ub = 2.5;
  float scale_lb = 0.2;
  float scale_ub = 2.0;

  TensorShape s_out({h_out, w_out, s_in[2]});

  auto op = std::make_unique<RandomCropAndResizeOp>(h_out, w_out, scale_lb, scale_ub, aspect_lb, aspect_ub);
  Status s;
  for (auto i = 0; i < 100; i++) {
    s = op->Compute(input_tensor_row, &output_tensor_row);
    EXPECT_TRUE(s.IsOk());
  }

  MS_LOG(INFO) << "RandomCropAndResizeOp simple test finished";
}
TEST_F(MindDataTestRandomCropAndResizeOp, TestOpSimpleTest2) {
  MS_LOG(INFO) << " starting RandomCropAndResizeOp simple test";
  TensorShape s_in = input_tensor_->shape();
  TensorRow input_tensor_row;
  input_tensor_row.push_back(input_tensor_);
  input_tensor_row.push_back(input_tensor_);
  TensorRow output_tensor_row;
  std::shared_ptr<Tensor> output_tensor;
  int h_out = 1024;
  int w_out = 2048;
  float aspect_lb = 1;
  float aspect_ub = 1.5;
  float scale_lb = 0.2;
  float scale_ub = 2.0;

  TensorShape s_out({h_out, w_out, s_in[2]});

  auto op = std::make_unique<RandomCropAndResizeOp>(h_out, w_out, scale_lb, scale_ub, aspect_lb, aspect_ub);
  Status s;
  for (auto i = 0; i < 100; i++) {
    s = op->Compute(input_tensor_row, &output_tensor_row);
    EXPECT_TRUE(s.IsOk());
  }

  MS_LOG(INFO) << "RandomCropAndResizeOp simple test finished";
}
TEST_F(MindDataTestRandomCropAndResizeOp, TestOpSimpleTest3) {
  MS_LOG(INFO) << " starting RandomCropAndResizeOp simple test";
  TensorShape s_in = input_tensor_->shape();
  TensorRow input_tensor_row;
  input_tensor_row.push_back(input_tensor_);
  input_tensor_row.push_back(input_tensor_);
  TensorRow output_tensor_row;
  std::shared_ptr<Tensor> output_tensor;
  int h_out = 1024;
  int w_out = 2048;
  float aspect_lb = 0.2;
  float aspect_ub = 3;
  float scale_lb = 0.2;
  float scale_ub = 2.0;

  TensorShape s_out({h_out, w_out, s_in[2]});

  auto op = std::make_unique<RandomCropAndResizeOp>(h_out, w_out, scale_lb, scale_ub, aspect_lb, aspect_ub);
  Status s;
  for (auto i = 0; i < 100; i++) {
    s = op->Compute(input_tensor_row, &output_tensor_row);
    EXPECT_TRUE(s.IsOk());
  }

  MS_LOG(INFO) << "RandomCropAndResizeOp simple test finished";
}