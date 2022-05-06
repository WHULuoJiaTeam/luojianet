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

#ifndef GDAL2CV_H_
#define GDAL2CV_H_

#include <iostream>
#include <vector>
#include <string>

#include <gdal_priv.h>
#include <gdal.h>
#include <opencv2/opencv.hpp>

using std::cout;
using std::string;
using std::vector;
using cv::Mat;
using cv::Scalar;

namespace luojianet_ms {
// Implement gdal dtype -> opencv dtype in luojianet_ms,
// inspired by https://github.com/HiKapok/GDAL2CV.
class GDAL2CV {
 public:
  GDAL2CV();
  ~GDAL2CV();

	/// \Data read by GDAL, and transferred to CV::Mat type.
	/// \param[in] filename, data filename.
	/// \param[in] xStart, the pixel offset to the top left corner of the region of the band to be accessed.
	/// \param[in] yStart, the line offset to the top left corner of the region of the band to be accessed.
	/// \param[in] xWidth, the width of the region of the band to be accessed in pixels.
	/// \param[in] yWidth, the height of the region of the band to be accessed in lines.
	/// \return out, data in cv::Mat dtype.
  Mat gdal_read(const string& filename, int xStart, int yStart, int xWidth, int yWidth);
};

}	// namespace luojianet_ms

#endif	// GDAL2CV_H_