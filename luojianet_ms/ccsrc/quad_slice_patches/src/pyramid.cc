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

#include "pyramid.h"

namespace luojianet_ms {

Pyramid::Pyramid() {}

Pyramid::~Pyramid() {}

void Pyramid::create_pyramid(Mat& image, Mat& label) {
	make_border(image, label);
	image_pyramid.push_back(image);
	label_pyramid.push_back(label);

	resize(image, image, Size(16, 16));
	resize(label, label, Size(16, 16), 0, 0, INTER_NEAREST);  // mode = 'inter_nearest' when resize the label.
	image_pyramid.push_back(image);
	label_pyramid.push_back(label);
}

void Pyramid::make_border(Mat& image, Mat& label) {
	int w = image.cols;
	int h = image.rows;
	int exponent = log(min(w, h)) / log(2);
	int size = pow(2.0, (float)exponent);

	while (size < w || size < h) {
		exponent += 1;
		size = pow(2.0, (float)exponent);
	}

	int pad_h = size - h;
	int pad_w = size - w;
	if (pad_h > 0 || pad_w > 0) {
		copyMakeBorder(image, image, 0, pad_h, 0, pad_w, BORDER_CONSTANT, Scalar::all(0));  // default image value to pad.
		copyMakeBorder(label, label, 0, pad_h, 0, pad_w, BORDER_CONSTANT, Scalar::all(255));  // default label value (ignore label) to pad.
	}
}

// Original method: data pyramid in kPyramidLevels.
//void Pyramid::create_pyramid(Mat& image, Mat& label) {
//	make_border(image, label);
//
//	image_pyramid.push_back(image);
//	label_pyramid.push_back(label);
//
//	const float ratio = 0.5f;
//	for (int level = 1; level < kPyramidLevels; level++) {
//		if (image.cols <= 512) {
//			break;
//		}
//		resize(image, image, Size(image.cols*ratio, image.rows*ratio));
//		// Mode = 'inter_nearest' when resize the label.
//		resize(label, label, Size(label.cols*ratio, label.rows*ratio), 0, 0, INTER_NEAREST);
//		image_pyramid.push_back(image);
//		label_pyramid.push_back(label);
//	}
//}

}  // namespace luojianet_ms