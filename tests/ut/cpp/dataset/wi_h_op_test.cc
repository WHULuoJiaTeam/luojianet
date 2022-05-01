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
#include "minddata/dataset/kernels/image/wi_h_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"

using namespace luojianet_ms::dataset;
using luojianet_ms::MsLogLevel::INFO;
using luojianet_ms::ExceptionType::NoExceptionType;
using luojianet_ms::LogStream;

class MindDataTestWI_HOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestWI_HOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

TEST_F(MindDataTestWI_HOp, TestOp1) {
  MS_LOG(INFO) << "Doing testWI_H.";
  std::unique_ptr<WI_HOp> op(new WI_HOp());
  EXPECT_TRUE(op->OneToOne());
  
  std::string band3 ="/test_luojianet/luojianet1/luojianet/test_data/LC81220392013285LGN01/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF";
  std::string band4 ="/test_luojianet/luojianet1/luojianet/test_data/LC81220392013285LGN01/LC08_L1TP_122039_20131012_20170429_01_T1_B4.TIF";
  std::string band6 ="/test_luojianet/luojianet1/luojianet/test_data/LC81220392013285LGN01/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF";

  cv::Mat input_band3 = cv::imread(band3, cv::IMREAD_LOAD_GDAL | cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
  cv::Mat input_band4 = cv::imread(band4, cv::IMREAD_LOAD_GDAL | cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
  cv::Mat input_band6 = cv::imread(band6, cv::IMREAD_LOAD_GDAL | cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
  
  cv::Mat mergedst;
  vector<cv::Mat> mergesrc;
  mergesrc.push_back(input_band3);
  mergesrc.push_back(input_band4);
  mergesrc.push_back(input_band6);
  cv::merge(mergesrc, mergedst);

  // create new tensor to test conversion
  std::shared_ptr<Tensor> test_input;
  std::shared_ptr<CVTensor> input_cv_tensor;
  CVTensor::CreateFromMat(mergedst, 3, &input_cv_tensor);
  test_input = std::dynamic_pointer_cast<Tensor>(input_cv_tensor);
  
  Status s = op->Compute(test_input, &output_tensor_);
  
  std::shared_ptr<CVTensor> output_cv = CVTensor::AsCVTensor(output_tensor_);
  cv::Mat output_img = output_cv->mat();
  cv::imwrite("/test_luojianet/luojianet1/luojianet/temp_wi_h.tif", output_img);
  MS_LOG(INFO) << "testWI_H end.";
  //EXPECT_EQ(s, Status::OK());
}

