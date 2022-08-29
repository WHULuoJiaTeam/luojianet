/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#ifndef PYRAMID_H_
#define PYRAMID_H_

#include <vector>

#include <opencv2/opencv.hpp>

using std::vector;
using cv::Mat;
using cv::Size;
using cv::resize;
using cv::Scalar;
using cv::min;
using cv::INTER_NEAREST;
using cv::BORDER_CONSTANT;

namespace luojianet_ms {

class Pyramid {
 public:
  Pyramid();
  ~Pyramid();

	/// \Create data pyramid in kPyramidLevels.
	/// \In order to accelerate processing and quad-tree search,
	/// \suppose that the default data size = 4096, directly resample data to 16 x 16.
	/// \param[in] image, cv::Mat type input image.
	/// \param[in] label, cv::Mat type input label.
  void create_pyramid(Mat& image, Mat& label);

	/// \Pad tha data to fit in the power-of-two for later quadtree search.
	/// \param[in] image, cv::Mat type input image.
	/// \param[in] label, cv::Mat type input label.
  void make_border(Mat& image, Mat& label);

	vector<Mat> get_image_pyramid() const { return image_pyramid; }
	vector<Mat> get_label_pyramid() const { return label_pyramid; }

 private:
  //const int kPyramidLevels = 6;

  vector<Mat> image_pyramid;
  vector<Mat> label_pyramid;
};

}  // luojianet_ms

#endif	// PYRAMID_H_