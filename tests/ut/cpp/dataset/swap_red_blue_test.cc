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
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/image/swap_red_blue_op.h"
#include "utils/log_adapter.h"

using namespace luojianet_ms::dataset;
using luojianet_ms::MsLogLevel::INFO;
using luojianet_ms::ExceptionType::NoExceptionType;
using luojianet_ms::LogStream;

class MindDataTestSwapRedBlueOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestSwapRedBlueOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

TEST_F(MindDataTestSwapRedBlueOp, TestOp1) {
  MS_LOG(INFO) << "Doing testSwapRedBlue.";
  // SwapRedBlue params
  std::unique_ptr<SwapRedBlueOp> op(new SwapRedBlueOp());
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor_);
  size_t actual = 0;
  if (s == Status::OK()) {
    actual = output_tensor_->shape()[0] * output_tensor_->shape()[1] * output_tensor_->shape()[2];
  }
  EXPECT_EQ(actual, input_tensor_->shape()[0] * input_tensor_->shape()[1] * 3);
  EXPECT_EQ(s, Status::OK());
}

