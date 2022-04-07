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

#include <iostream>
#include <vector>

#include <gdal_priv.h>
#include <gdal.h>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "gdal2cv.h"
#include "blockread.h"
#include "pyramid.h"
#include "quadtree.h"

namespace py = pybind11;
using std::cout;
using std::vector;
using cv::Mat;

namespace luojianet_ms {

/// \Get init information of big_input data by using gdal lib.
///
/// \param[in] image_path, big_input image path.
/// \param[in] init_cols, init cols of big_input image.
/// \param[in] init_rows, init rows of big_input image.
/// \param[in] init_bands, init num bands of big_input image.
void get_init_cols_rows(const string& image_path, int* init_cols, int* init_rows, int* init_bands) {
	const char* imagepath = image_path.data();
	GDALAllRegister();
	GDALDataset* poSrc = (GDALDataset*)GDALOpen(imagepath, GA_ReadOnly);
	if (poSrc == NULL) {
		cout << "GDAL failed to open " << image_path;
		return;
	}
	*init_cols = poSrc->GetRasterXSize();
	*init_rows = poSrc->GetRasterYSize();
	*init_bands = poSrc->GetRasterCount();
	GDALClose((GDALDatasetH)poSrc);
	poSrc = NULL;
}

/// \Only choose the first three bands of an image.
/// \Todo: The support for band-selection algorithm, especially for high-spectral data.
///
/// \param[in] image_path, image path.
/// \param[in] col_cord, init read cord of image cols.
/// \param[in] row_cord, init read cord of image rows.
/// \param[in] current_block_cols, basic processing unit of image cols.
/// \param[in] current_block_rows, basic processing unit of image rows.
/// \return three-band image, cv::Mat dtype.
Mat band_selection(const string& image_path,
				   int col_cord, int row_cord,
		           int current_block_cols, int current_block_rows) {
	GDAL2CV gdal2cv;
	Mat image = gdal2cv.gdal_read(image_path, col_cord, row_cord, current_block_cols, current_block_rows);

	vector<Mat> all_channels;
	vector<Mat> three_channels;
	split(image, all_channels);
	for (int i = 0; i < 3; i++) {
		three_channels.push_back(all_channels.at(i));
	}
	merge(three_channels, image);
	return image;
}

/// \One channel data, Mat->Numpy.
///
/// \param[in] input, cv::Mat data.
/// \return dst, numpy dtype.
py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(Mat& input) {
	py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows,input.cols }, input.data);
	return dst;
}

/// \Three channel data, Mat->Numpy.
///
/// \param[in] input, cv::Mat data.
/// \return dst, numpy dtype.
py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(Mat& input) {
	py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows, input.cols, 3 }, input.data);
	return dst;
}

/// \Get the minimum bounding rectangle data of one-specified class.
///
/// \param[in] device_num, the number of device for training.
/// \param[in] rank_id, the current device ID.
/// \param[in] image_path, big_input image path.
/// \param[in] label_path, big_input label path.
/// \param[in] n_classes, num classes of ground features.
/// \param[in] ignore_label, pad value of ground features.
/// \param[in] block_size, basic processing unit.
/// \param[in] max_searchsize, max output data size.
/// \param[in] min_searchsize, min output data size.
/// \return out, image-label objects in Numpy dtype.
py::list get_objects(const int device_num, const int rank_id,
	                 const string& image_path, const string& label_path,
	                 const int n_classes, const int ignore_label,
	                 const int block_size, const int max_searchsize, const int min_searchsize) {
	// Struct for store export data.
	typedef struct {
		vector<Mat> image_objects;
		vector<Mat> label_objects;
	} Object;
	Object object;

	// Get init information of big_input.
	int init_cols, init_rows, init_bands = 0;
	get_init_cols_rows(image_path, &init_cols, &init_rows, &init_bands);

	Mat image, label;
	if (init_cols <= block_size && init_rows <= block_size) {
		// Todo: The support for high-spectral data (more than 3 bands).
		if (init_bands > 3) {
			image = band_selection(image_path, 0, 0, init_cols, init_rows);
		}
		else {
			image = cv::imread(image_path, -1);
		}
		label = cv::imread(label_path, -1);
		// Create data pyramid.
		Pyramid pyramid;
		pyramid.create_pyramid(image, label);
		Mat top_level_image = pyramid.image_pyramid.back();
		Mat top_level_label = pyramid.label_pyramid.back();
		Mat ori_level_image = pyramid.image_pyramid.front();
		Mat ori_level_label = pyramid.label_pyramid.front();
		// Create quadtree search tree.
		QuadTree quadtree;
		quadtree.random_search(top_level_image, top_level_label, n_classes, ignore_label);
		quadtree.get_multiscale_object(ori_level_image, ori_level_label, max_searchsize, min_searchsize);
		// Store the output data objects.
		object.image_objects = quadtree.image_objects;
		object.label_objects = quadtree.label_objects;
	}
	else {
		// The cord of all-class related data blocks are sequentially stored in related_block_cord.
		BlockRead blockread;
		blockread.get_related_block(label_path, init_cols, init_rows, n_classes, ignore_label, block_size);
		vector<Vector2> related_block_cord = blockread.related_block_cord;
		// The data blocks are sequentially processed depend on device num and rank_id of current device.
		int num_class_related_block_cord = related_block_cord.size();
		int num_one_device_related_block_cord = num_class_related_block_cord / device_num;
		int num_residual_related_block_cord = num_class_related_block_cord % device_num;
		int sequence_beg_index = 0;
		// The last device is responsible to process residual data blocks.
		if (num_residual_related_block_cord > 0) {
			if (rank_id == device_num - 1) {
				sequence_beg_index = rank_id * num_one_device_related_block_cord;
				num_one_device_related_block_cord = 
					num_class_related_block_cord - (rank_id * num_one_device_related_block_cord);
			}
		}
		else {
			sequence_beg_index = rank_id * num_one_device_related_block_cord;
		}
		// for each class-related data block, read their original cord in an sequence.
		for (int i = 0; i < num_one_device_related_block_cord; i++) {
			int row_cord = related_block_cord[sequence_beg_index + i].x;
			int col_cord = related_block_cord[sequence_beg_index + i].y;
			int current_block_rows = block_size;
			int current_block_cols = block_size;
			// Process the residual data block of big_input data.
			if (row_cord + block_size > init_rows) {
				current_block_rows = init_rows - row_cord;
			}
			if (col_cord + block_size > init_cols) {
				current_block_cols = init_cols - col_cord;
			}
			GDAL2CV gdal2cv;
			if (init_bands > 3) {
				image = band_selection(image_path, col_cord, row_cord, current_block_cols, current_block_rows);
			}
			else {
				image = gdal2cv.gdal_read(image_path, col_cord, row_cord, current_block_cols, current_block_rows);
			}
			label = gdal2cv.gdal_read(label_path, col_cord, row_cord, current_block_cols, current_block_rows);
			// Create data pyramid.
			Pyramid pyramid;
			pyramid.create_pyramid(image, label);
			Mat top_level_image = pyramid.image_pyramid.back();
			Mat top_level_label = pyramid.label_pyramid.back();
			Mat ori_level_image = pyramid.image_pyramid.front();
			Mat ori_level_label = pyramid.label_pyramid.front();
			// Create quadtree search tree.
			QuadTree quadtree;
			quadtree.random_search(top_level_image, top_level_label, n_classes, ignore_label);
			quadtree.get_multiscale_object(ori_level_image, ori_level_label, max_searchsize, min_searchsize);
			// Store the output data objects.
			object.image_objects.insert(object.image_objects.end(),
										quadtree.image_objects.begin(), quadtree.image_objects.end());
			object.label_objects.insert(object.label_objects.end(),
										quadtree.label_objects.begin(), quadtree.label_objects.end());
		}
	}
	// If the output data objects is empty.
	if (object.image_objects.empty()) {
		GDAL2CV gdal2cv;
		Mat image = gdal2cv.gdal_read(image_path, 0, 0, 1024, 1024);
		Mat label = gdal2cv.gdal_read(label_path, 0, 0, 1024, 1024);
		object.image_objects.push_back(image);
		object.label_objects.push_back(label);
	}
	// Convert the Mat dtype to Numpy dtype.
	Mat src_image, src_label;
	py::array_t<unsigned char> dst_image, dst_label;
	py::list out_image_objects, out_label_objects;
	for (int index = 0; index < object.image_objects.size(); index++) {
		// image objects.
		src_image = object.image_objects[index];
		dst_image = cv_mat_uint8_3c_to_numpy(src_image);
		out_image_objects.append(dst_image);
		// label objects.
		src_label = object.label_objects[index];
		dst_label = cv_mat_uint8_1c_to_numpy(src_label);
		out_label_objects.append(dst_label);
	}
	py::list out;
	out.append(out_image_objects);
	out.append(out_label_objects);
	return out;
}

// Python API.
PYBIND11_MODULE(geobject, m) {
	m.doc() = "input filename, output py::list for objects";
	// Add bindings function.
	m.def("get_objects", &get_objects);
}

}	// namespace luojianet_ms