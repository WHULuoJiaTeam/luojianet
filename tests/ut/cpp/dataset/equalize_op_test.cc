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
#include "minddata/dataset/kernels/image/equalize_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"

using namespace luojianet_ms::dataset;
using luojianet_ms::LogStream;
using luojianet_ms::ExceptionType::NoExceptionType;
using luojianet_ms::MsLogLevel::INFO;

class MindDataTestEqualizeOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestEqualizeOp() : CVOpCommon() {}
};

TEST_F(MindDataTestEqualizeOp, TestOp1) {
  MS_LOG(INFO) << "Doing testEqualizeOp.";

  std::shared_ptr<Tensor> output_tensor;
  std::unique_ptr<EqualizeOp> op(new EqualizeOp());
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  CheckImageShapeAndData(output_tensor, kEqualize);
}
