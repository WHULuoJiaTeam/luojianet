/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/resize_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestResizeOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestResizeOp() : CVOpCommon() {}
};

TEST_F(MindDataTestResizeOp, TestOp) {
  MS_LOG(INFO) << "Doing testResize";
  // Resizing with a factor of 0.5
  TensorShape s = input_tensor_->shape();
  int output_w = 0.5 * s[0];
  int output_h = (s[0] * output_w) / s[1];
  std::shared_ptr<Tensor> output_tensor;
  // Resizing
  std::unique_ptr<ResizeOp> op(new ResizeOp(output_h, output_w));
  Status st = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(st.IsOk());
  CheckImageShapeAndData(output_tensor, kResizeBilinear);
  MS_LOG(INFO) << "testResize end.";
}
