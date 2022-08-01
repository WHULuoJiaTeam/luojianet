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

#ifndef BLOCKREAD_H_
#define BLOCKREAD_H_

#include <string>
#include <vector>

#include "gdal2cv.h"
#include "pyramid.h"
#include "boundbox.h"

using std::vector;
using std::cout;
using std::pair;
using cv::Mat;
using cv::Mat_;
using cv::Scalar;
using cv::Size;
using cv::Rect;
using cv::Point;
using cv::INTER_NEAREST;
using cv::BORDER_CONSTANT;
using cv::COLOR_BGR2GRAY;
using cv::countNonZero;

namespace luojianet_ms {

class BlockRead {
 public:
	BlockRead();
	~BlockRead();

	/// \Get the label-value realted data block (default size is 4096 x 4096).
  /// \param[in] image_path, image filename.
	/// \param[in] label_path, label filename.
	/// \param[in] init_cols, init cols of big_input image.
	/// \param[in] init_rows, init rows of big_input image.
	/// \param[in] n_classes, num classes of labels.
	/// \param[in] ignore_label, pad value of ground features.
	/// \param[in] block_size, basic processing unit for big_input data.
	//void get_related_block(string& label_path, int init_cols, int init_rows, int n_classes, int ignore_label, int block_size);
  void get_related_block(string& image_path, string& label_path,
                         int init_cols, int init_rows, int n_classes, int ignore_label, int block_size,
                         int max_searchsize);

	/// \Get class_attribute matrix based on label-value.
  /// \param[in] image_path, image filename.
	/// \param[in] label_path, label filename.
	/// \param[in] init_cols, init cols of big_input image.
	/// \param[in] init_rows, init rows of big_input image.
	/// \param[in] n_classes, num classes of labels.
	/// \param[in] ignore_label, pad value.
	/// \param[in] block_size, basic processing unit for big_input data.
	//void get_class_attribute(string& label_path, int init_cols, int init_rows, int n_classes, int ignore_label, int block_size);
	void get_class_attribute(string& image_path, string& label_path,
                           int init_cols, int init_rows, int n_classes, int ignore_label, int block_size,
                           int max_searchsize);

	/// \In order to accelerate processing and quad-tree search,
	/// \suppose that the default data size = 4096, directly resample data to 16 x 16 for label-value quick statistic.
	/// \param[in] label, cv::Mat label data.
	/// \param[in] block_index, the index of processing unit for big_input data.
	/// \param[in] n_classes, num classes of labels.
	/// \param[in] ignore_label, the value to skip over in statistic.
	void quick_statistic_class(Mat& label, int block_index, int n_classes, int ignore_label);

	/// \In resampled label, compare the block processing unit to its 8 neighbours,
	/// \find the same label-value block and save the class_attribute matrix.
	/// \param[in] init_cols, init cols of big_input image.
	/// \param[in] init_rows, init rows of big_input image.
	/// \param[in] block_size, basic processing unit for big_input data.
	void search_related_block(int init_cols, int init_rows, int block_size);

	/// \For each class, get the original cord of its related block,
	/// \and saved it sequencetly in vector.
	/// \param[in] block_size, basic processing unit for big_input data.
	void store_related_blockcord(int block_size);

  // Get slice patches envenly from original data in cv::Mat datatype.
  /// \param[in] image, image size is (block_size, block_size).
  /// \param[in] label, label size is (block_size, block_size).
  /// \param[in] max_searchsize, patch size is equal to max_searchsize.
  //void get_slice_patches(Mat& image, Mat& label, int max_searchsize);

	vector<Vector2> get_related_block_cord() const { return related_block_cord; }

  //vector<Mat> get_image_patches() const { return image_patches; }
  //vector<Mat> get_label_patches() const { return label_patches; }

 private:
	Mat_<uchar> class_attribute;  // class_attribute matrix.
	
	vector<Mat_<uchar>> related_class_mask;  // each label class has an class_attribute matrix.

	vector<Vector2> related_block_cord;  	// cord vector of class realted block.

  //vector<Mat> image_patches;  // random_search image results.
  //vector<Mat> label_patches;  // random_search label results.
};

}  // namespace luojianet_ms

#endif	// BLOCKREAD_H_