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

#include "blockread.h"
#include "gdal2cv.h"

using namespace luojianet_ms;


BlockRead::BlockRead() {

}


BlockRead::~BlockRead() {

}


void BlockRead::get_related_block(const string &label_path, int init_cols, int init_rows, int n_classes, int ignore_label, int block_size) {
	get_class_attribute(label_path, init_cols, init_rows, n_classes, ignore_label, block_size);
	search_related_block(init_cols, init_rows, block_size);
	store_related_blockcord(block_size);
}


Vector2 maskcord2imgcord(Vector2 &mask_lowerbound_cord, int block_size) {
	Vector2 img_lowerbound_cord = mask_lowerbound_cord * block_size;
	return img_lowerbound_cord;
}


void BlockRead::store_related_blockcord(int block_size) {
	/* For each class, get the related block boundary. */
	for (vector<cv::Mat_<uchar>>::iterator it = related_class_mask.begin(); it != related_class_mask.end(); it++) {
		for (int i = 0; i < (*it).rows; i++) {
			for (int j = 0; j < (*it).cols; j++) {
				if ((*it)(i, j) == true) {
					Vector2 mask_lowerbound_cord(i, j);
					Vector2 img_lowerbound_cord = maskcord2imgcord(mask_lowerbound_cord, block_size);

					/* Put all-class related block lowerbound in one std::vector, so it can be read in an sequence. */
					related_block_cord.push_back(img_lowerbound_cord);
				}
			}
		}
	}
	if (related_block_cord.empty()) {
		cout << "Failed to store related blockcord.";
		return;
	}
}


cv::Mat make_label_border(cv::Mat &label, int block_size) {
	int w = label.cols;
	int h = label.rows;
	int pad_h = block_size - h;
	int pad_w = block_size - w;
	if (pad_h > 0 || pad_w > 0) {
		cv::Scalar ignore_value = cv::Scalar(255, 255, 255);
		copyMakeBorder(label, label, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, ignore_value);
	}
	return label;
}


/*

class_attribute matrix:

block_index	    class1       class2    ...
0             true/false   true/false
1             true/false   true/false
...

*/
void BlockRead::get_class_attribute(const string &label_path, int init_cols, int init_rows, int n_classes, int ignore_label, int block_size) {
	int block_num = ceil((float)init_cols / (float)block_size) * ceil((float)init_rows / (float)block_size);
	cv::Mat_<uchar> all_class_attribute(block_num, n_classes, uchar(0));
	all_class_attribute.copyTo(class_attribute);
	int block_index = 0;

	GDAL2CV gdal2cv;
	for (int i = 0; i < init_rows; i += block_size) {
		for (int j = 0; j < init_cols; j += block_size) {
			int current_block_rows = block_size;
			int current_block_cols = block_size;
			if (i + block_size > init_rows) {
				current_block_rows = init_rows - i;
			}
			if (j + block_size > init_cols) {
				current_block_cols = init_cols - j;
			}
			
			cv::Mat label = gdal2cv.gdal_read(label_path, j, i, current_block_cols, current_block_rows);

			/* Make border the residule block to standard BLOCK_SIZE for quick statistic in 512��512 size. */
			if (label.rows < block_size || label.cols < block_size) {
				cv::Mat label_border = make_label_border(label, block_size);
				quick_statistic_class(label_border, block_index, n_classes, ignore_label);
			}
			else {
				quick_statistic_class(label, block_index, n_classes, ignore_label);
			}
			block_index++;
		}
	}
	//cv::imwrite("class_attribute.tif", class_attribute);
}


void BlockRead::quick_statistic_class(cv::Mat &label, int block_index, int n_classes, int ignore_label) {
	int value = 0;
	vector<int> label_value(n_classes, 0);

	/* Mode = 'inter_nearest' when resize the label. */
	cv::resize(label, label, cv::Size(512, 512), 0, 0, cv::INTER_NEAREST);
	for (int i = 0; i < label.rows; i++) {
		for (int j = 0; j < label.cols; j++) {
			value = label.at<uchar>(i, j);
			if (value == ignore_label) {
				continue;
			}
			label_value[value]++;
		}
	}

	for (int k = 0; k < (int)label_value.size(); k++) {
		if (label_value[k] > 0) {
			class_attribute(block_index, k) = true;
		}
	}
}


void push(vector<int> &search_results, int &row, int &col) {
	search_results.push_back(row);
	search_results.push_back(col);
}


bool pop(vector<int>& search_results, int &row, int &col) {
	if (search_results.size() < 2) {
		return false;
	}

	col = search_results.back();
	search_results.pop_back();
	row = search_results.back();
	search_results.pop_back();

	return true;
}


void BlockRead::search_related_block(int init_cols, int init_rows, int block_size) {
	int n_classes = class_attribute.cols;
	int block_num = class_attribute.rows;
	int block_num_inrows = ceil((float)init_rows / (float)block_size);
	int block_num_incols = ceil((float)init_cols / (float)block_size);

	/* Search related blocks for each classes. */
	for (int i = 0; i < n_classes; i++) {
		int init_search_class = -1;
		int init_search_block = -1;
		for (int j = 0; j < block_num; j++) {
			if (class_attribute(j, i) == true) {
				init_search_class = i;
				init_search_block = j;
				break;
			}
		}
		int init_search_row = init_search_block / block_num_incols;
		int init_search_col = init_search_block % block_num_incols;

		const int dx8[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
		const int dy8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
		cv::Mat_<uchar> istaken(block_num_inrows, block_num_incols, uchar(0));

		/* If the resize 512x512 block have all-false-atrribute class, push back zero-matrix. */
		if (init_search_class == -1 || init_search_block == -1) {
			related_class_mask.push_back(istaken);
			continue;
		}

		int row = init_search_row;
		int col = init_search_col;
		vector<int> search_results;
		push(search_results, row, col);
		while (pop(search_results, row, col)) {
			istaken(row, col) = true;
			for (int i = 0; i < 8; i++) {
				int dxrow = row + dx8[i];
				int dycol = col + dy8[i];
				if (dxrow >= 0 && dxrow < block_num_inrows && dycol >= 0 && dycol < block_num_incols && istaken(dxrow, dycol) == false) {
					int current_block_index = dxrow * block_num_incols + dycol;
					/* If current block index has true attribute for current class. */
					if (class_attribute(current_block_index, init_search_class) == true) {
						push(search_results, dxrow, dycol);
					}
				}
			}
		}
		//cv::imwrite("block_search.tif", istaken);
		related_class_mask.push_back(istaken);
	}
}