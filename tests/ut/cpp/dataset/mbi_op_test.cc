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

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common/common.h"
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/image/mbi_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"

using namespace luojianet_ms::dataset;
using luojianet_ms::MsLogLevel::INFO;
using luojianet_ms::ExceptionType::NoExceptionType;
using luojianet_ms::LogStream;

class MindDataTestMBIOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestMBIOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

TEST_F(MindDataTestMBIOp, TestOp1) {
  MS_LOG(INFO) << "Doing testMBI.";
  std::unique_ptr<MBIOp> op(new MBIOp(3, 20, 1));
  EXPECT_TRUE(op->OneToOne());
  
  std::string folder_path = "/test_luojianet/luojianet1/luojianet/building.tiff";
  // prepare 3 channel image
  cv::Mat input_img = cv::imread(folder_path);
  // create new tensor to test conversion
  std::shared_ptr<Tensor> test_input;
  std::shared_ptr<CVTensor> input_cv_tensor;
  CVTensor::CreateFromMat(input_img, 3, &input_cv_tensor);
  test_input = std::dynamic_pointer_cast<Tensor>(input_cv_tensor);
  
  Status s = op->Compute(test_input, &output_tensor_);
  
  std::shared_ptr<CVTensor> output_cv = CVTensor::AsCVTensor(output_tensor_);
  cv::Mat output_img = output_cv->mat();
  cv::imwrite("/test_luojianet/luojianet1/luojianet/building_mbi.tif", output_img);
  MS_LOG(INFO) << "testMBI end.";
  //EXPECT_EQ(s, Status::OK());
}

