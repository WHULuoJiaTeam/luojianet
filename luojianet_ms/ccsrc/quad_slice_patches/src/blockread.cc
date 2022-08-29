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

namespace luojianet_ms {

BlockRead::BlockRead() {}

BlockRead::~BlockRead() {}

//void BlockRead::get_related_block(string& label_path, int init_cols, int init_rows, int n_classes, int ignore_label, int block_size) {
//  get_class_attribute(label_path, init_cols, init_rows, n_classes, ignore_label, block_size);
//  search_related_block(init_cols, init_rows, block_size);
//  store_related_blockcord(block_size);
//}

void BlockRead::get_related_block(string& image_path, string& label_path,
                                  int init_cols, int init_rows, int n_classes, int ignore_label, int block_size,
                                  int max_searchsize) {
	get_class_attribute(image_path, label_path, init_cols, init_rows, n_classes, ignore_label, block_size, max_searchsize);
	search_related_block(init_cols, init_rows, block_size);
	store_related_blockcord(block_size);
}

//// Function used in 'get_class_attribute'.
//Mat make_label_border(Mat& label, int block_size) {
//  int w = label.cols;
//  int h = label.rows;
//  int pad_h = block_size - h;
//  int pad_w = block_size - w;
//  if (pad_h > 0 || pad_w > 0) {
//    copyMakeBorder(label, label, 0, pad_h, 0, pad_w, BORDER_CONSTANT, Scalar::all(255));
//  }
//  return label;
//}

// Function used in 'get_class_attribute'.
pair<Mat, Mat> make_border(Mat& image, Mat& label, const int block_size) {
  pair<Mat, Mat> border_result;
	int w = image.cols;
	int h = image.rows;
	int pad_h = block_size - h;
	int pad_w = block_size - w;
	if (pad_h > 0 || pad_w > 0) {
    copyMakeBorder(image, image, 0, pad_h, 0, pad_w, BORDER_CONSTANT, Scalar::all(0));  // default image value to pad.
		copyMakeBorder(label, label, 0, pad_h, 0, pad_w, BORDER_CONSTANT, Scalar::all(255));  // default label value (ignore label) to pad.
	}
  border_result.first = image;
  border_result.second = label;
	return border_result;
}

// Function used in 'get_class_attribute'.
void BlockRead::quick_statistic_class(Mat& label, int block_index, int n_classes, int ignore_label) {
	int value = 0;
	vector<int> label_value(n_classes, 0);

	// For each block (4096 x 4096), resampled to 16 x 16 for quick statistic.
	cv::resize(label, label, Size(16, 16), 0, 0, INTER_NEAREST);  // Mode = 'inter_nearest' when resize the label.
	for (int i = 0; i < label.rows; i++) {
		for (int j = 0; j < label.cols; j++) {
			value = label.at<uchar>(i, j);
			if (value == ignore_label) {
				continue;
			}
			label_value[value]++;
		}
	}
	/// label_value:
	///   class1       class2    ...
	/// pixel num    pixel num
	///    ...
	for (int k = 0; k < (int)label_value.size(); k++) {
		if (label_value[k] > 0) {
			class_attribute(block_index, k) = true;
		}
	}
}

//// Funtion used in 'get_class_attribute'.
//void BlockRead::get_slice_patches(Mat& image, Mat& label, int max_searchsize) {
//  // Ignore zero-matrix block_size data.
//  Mat gray_image;
//  cvtColor(image, gray_image, COLOR_BGR2GRAY);
//  if (cv::countNonZero(gray_image) < 1) {
//    return;
//  }
//
//  int rows = image.rows;
//  int cols = image.cols;
//  const int patch_size = max_searchsize;
//  for (int i = 0; i < rows; i += patch_size) {
//    for (int j = 0; j < cols; j += patch_size) {
//      int current_row_patch_size = patch_size;
//      int current_col_patch_size = patch_size;
//      if (i + patch_size > rows) {
//        current_row_patch_size = rows - i;
//      }
//      if (j + patch_size > cols) {
//        current_col_patch_size = cols - j;
//      }
//
//      Rect roi(j, i, current_col_patch_size, current_row_patch_size);
//      Mat image_patch = image(roi);
//      Mat label_patch = label(roi);
//
//      // If need, make border the data patch, to size (max_searchsize, max_searchsize).
//      pair<Mat, Mat> patch_border_result = make_border(image_patch, label_patch, patch_size);
//
//      // Ignore poor-information matrix.
//      Mat gray_image_patch;
//      cvtColor(patch_border_result.first, gray_image_patch, COLOR_BGR2GRAY);
//      double info_ratio = (double)countNonZero(gray_image_patch) / (double)(gray_image_patch.rows * gray_image_patch.cols);
//      if (info_ratio < 0.1) {
//        continue;
//      }
//
//      // For numpy, convert datatype to the CV_8U.
//      Mat image_patch_8UC3(patch_size, patch_size, CV_8UC3);
//      patch_border_result.first.copyTo(image_patch_8UC3);
//
//      Mat label_patch_8UC1(patch_size, patch_size, CV_8UC1);
//      patch_border_result.second.copyTo(label_patch_8UC1);
//
//      image_patches.push_back(image_patch_8UC3);
//      label_patches.push_back(label_patch_8UC1);
//    }
//  }
//}

/// Get the class attribute matrix of bif_input data:
/// block_index	    class1       class2    ...
///      0        true/false   true/false
///      1        true/false   true/false
///     ...
//void BlockRead::get_class_attribute(string& label_path, int init_cols, int init_rows, int n_classes, int ignore_label, int block_size) {
//	int block_num = ceil((float)init_cols / (float)block_size) * ceil((float)init_rows / (float)block_size);
//	Mat_<uchar> all_class_attribute(block_num, n_classes, uchar(0));
//	all_class_attribute.copyTo(class_attribute);
//
//	int block_index = 0;
//	GDAL2CV gdal2cv;
//	for (int i = 0; i < init_rows; i += block_size) {
//		for (int j = 0; j < init_cols; j += block_size) {
//			int current_block_rows = block_size;
//			int current_block_cols = block_size;
//			if (i + block_size > init_rows) {
//				current_block_rows = init_rows - i;
//			}
//			if (j + block_size > init_cols) {
//				current_block_cols = init_cols - j;
//			}
//			Mat label = gdal2cv.gdal_read(label_path, j, i, current_block_cols, current_block_rows);
//
//			// Make border the residule block to standard BLOCK_SIZE for quick statistic.
//			if (label.rows < block_size || label.cols < block_size) {
//				Mat label_border = make_label_border(label, block_size);
//				quick_statistic_class(label_border, block_index, n_classes, ignore_label);
//			}
//			else {
//				quick_statistic_class(label, block_index, n_classes, ignore_label);
//			}
//			block_index++;
//		}
//	}
//	//cv::imwrite("class_attribute.tif", class_attribute);
//}

/// Get the class attribute matrix of bif_input data:
/// block_index	    class1       class2    ...
///      0        true/false   true/false
///      1        true/false   true/false
///     ...
void BlockRead::get_class_attribute(string& image_path, string& label_path,
                                    int init_cols, int init_rows, int n_classes, int ignore_label, int block_size,
                                    int max_searchsize) {
  int block_num = ceil((float)init_cols / (float)block_size) * ceil((float)init_rows / (float)block_size);
  Mat_<uchar> all_class_attribute(block_num, n_classes, uchar(0));
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

      Mat image = gdal2cv.gdal_read(image_path, i, j, current_block_rows, current_block_cols);
      Mat label = gdal2cv.gdal_read(label_path, i, j, current_block_rows, current_block_cols);
  
  		// Make border the residule block to standard BLOCK_SIZE for quick statistic.
  		if (image.rows < block_size || image.cols < block_size) {
        pair<Mat, Mat> border_result = make_border(image, label, block_size);
        Mat image_border = border_result.first;
  			Mat label_border = border_result.second;

        // Get slice pathces from block data.
        //get_slice_patches(image_border, label_border, max_searchsize);

  			quick_statistic_class(label_border, block_index, n_classes, ignore_label);

  		}
  		else {
        // Get slice pathces from block data.
        //get_slice_patches(image, label, max_searchsize);

  			quick_statistic_class(label, block_index, n_classes, ignore_label);
  		}
  		block_index++;
  	}
  }
  //cv::imwrite("class_attribute.tif", class_attribute);
}

// Function used in 'search_related_block' for 8-neighbours search.
void push(vector<int>& search_results, int& row, int& col) {
	search_results.push_back(row);
	search_results.push_back(col);
}

// Function used in 'search_related_block' for 8-neighbours search.
bool pop(vector<int>& search_results, int& row, int& col) {
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

	// For each class, search its label-value related blocks.
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

		// Init search block.
		int init_search_row = init_search_block / block_num_incols;
		int init_search_col = init_search_block % block_num_incols;
		const int dx8[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
		const int dy8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
		Mat_<uchar> istaken(block_num_inrows, block_num_incols, uchar(0));

		// If the resampled 16 x 16 block have all-false-atrribute class, push back zero-matrix.
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
					// If current block index has true attribute for current label class.
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

// Function used in 'store_related_blockcord', mask cord -> image cord (bottom and left cord).
Vector2 maskcord2imgcord(Vector2& mask_lowerbound_cord, int block_size) {
	Vector2 img_lowerbound_cord = mask_lowerbound_cord * block_size;
	return img_lowerbound_cord;
}

// Funtion used in 'get_related_block'.
void BlockRead::store_related_blockcord(int block_size) {
	// For each class, get the related block cord.
	for (vector<Mat_<uchar>>::iterator it = related_class_mask.begin(); it != related_class_mask.end(); it++) {
		for (int i = 0; i < (*it).rows; i++) {
			for (int j = 0; j < (*it).cols; j++) {
				if ((*it)(i, j) == true) {
					Vector2 mask_lowerbound_cord(i, j);
					Vector2 img_lowerbound_cord = maskcord2imgcord(mask_lowerbound_cord, block_size);
					// Put all-class related block cord (bottom and left) in one std::vector, so it can be read in an sequence.
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

}  // namespace luojianet_ms