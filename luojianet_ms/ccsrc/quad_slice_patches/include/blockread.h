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

#ifndef BLOCKREAD_H
#define BLOCKREAD_H

#include <string>

#include "gdal2cv.h"
#include "pyramid.h"
#include "boundbox.h"

using namespace std;


class BlockRead {
public:
	cv::Mat_<uchar> class_attribute;
	vector<cv::Mat_<uchar>> related_class_mask;
	vector<Vector2> related_block_cord;
public:
	BlockRead();
	~BlockRead();

	void get_related_block(const string &label_path, int init_cols, int init_rows, int n_classes, int ignore_label, int block_size);

	void get_class_attribute(const string &label_path, int init_rows, int init_cols, int n_classes, int ignore_label, int block_size);
	void quick_statistic_class(cv::Mat &label, int block_index, int n_classes, int ignore_label);
	void search_related_block(int init_cols, int init_rows, int block_size);
	void store_related_blockcord(int block_size);
};

#endif