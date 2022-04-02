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

#ifndef PYRAMID_H
#define PYRAMID_H

#include <opencv2/opencv.hpp>

using namespace std;

// Pyramid settings
#define NLEVELS 6


class Pyramid {
public:
	vector<cv::Mat> image_pyramid;
	vector<cv::Mat> label_pyramid;
public:
	Pyramid();
	~Pyramid();

	void create_pyramid(cv::Mat &image, cv::Mat &label);
	void make_border(cv::Mat &image, cv::Mat &label);
};


#endif