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

#include <fstream>
//#include <direct.h>
#include <unistd.h>

#include "gdal2cv.h"
#include "blockread.h"
#include "pyramid.h"
#include "quadtree.h"
#include "json11.h"
#include "stringreplacer.h"

namespace py = pybind11;
using std::cout;
using std::vector;
using std::pair;
using cv::Mat;
using cv::Size;

using std::list;

using std::ofstream;
using std::to_string;
using std::endl;

namespace luojianet_ms {

/// \Get init information of big_input data by using gdal lib.
///
/// \param[in] image_path, big_input image path.
/// \param[in] init_cols, init cols of big_input image.
/// \param[in] init_rows, init rows of big_input image.
/// \param[in] init_bands, init num bands of big_input image.
//void get_init_cols_rows(string& image_path, int* init_cols, int* init_rows, int* init_bands) {
//	const char* imagepath = image_path.data();
//	GDALAllRegister();
//	GDALDataset* poSrc = (GDALDataset*)GDALOpen(imagepath, GA_ReadOnly);
//	if (poSrc == NULL) {
//		cout << "GDAL failed to open " << image_path;
//		return;
//	}
//	*init_cols = poSrc->GetRasterXSize();
//	*init_rows = poSrc->GetRasterYSize();
//	*init_bands = poSrc->GetRasterCount();
//	GDALClose((GDALDatasetH)poSrc);
//	poSrc = NULL;
//}

/// \Only choose the first three bands of an image.
/// \Todo: Support for band-selection algorithm, especially for high-spectral data.
///
/// \param[in] image_path, image path.
/// \param[in] col_cord, init read cord of image cols.
/// \param[in] row_cord, init read cord of image rows.
/// \param[in] current_block_cols, basic processing unit of image cols.
/// \param[in] current_block_rows, basic processing unit of image rows.
/// \return three-band image, cv::Mat dtype.
//Mat band_selection(string& image_path, int col_cord, int row_cord, int current_block_cols, int current_block_rows) {
//	GDAL2CV gdal2cv;
//	Mat image = gdal2cv.gdal_read(image_path, col_cord, row_cord, current_block_cols, current_block_rows);
//
//	vector<Mat> all_channels;
//	vector<Mat> three_channels;
//	split(image, all_channels);
//	for (int i = 0; i < 3; i++) {
//		three_channels.push_back(all_channels.at(i));
//	}
//	merge(three_channels, image);
//	return image;
//}

///// \One channel data, Mat->Numpy.
/////
///// \param[in] input, cv::Mat data.
///// \return dst, numpy dtype.
//py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(Mat& input) {
//	py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows,input.cols }, input.data);
//	return dst;
//}
//
///// \Three channel data, Mat->Numpy.
/////
///// \param[in] input, cv::Mat data.
///// \return dst, numpy dtype.
//py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(Mat& input) {
//	py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows, input.cols, 3 }, input.data);
//	return dst;
//}

// Get info of slice patches, put them into 'Patch' data structure.
void get_init_cols_rows(string& image_path, int* init_cols, int* init_rows, int* init_bands) {
  const char* imagepath = image_path.data();
  GDALAllRegister();
  GDALDataset* poSrc = (GDALDataset*)GDALOpen(imagepath, GA_ReadOnly);
  if (poSrc == NULL) {
    cout << "GDAL failed to open " << image_path;
    return;
  }
  *init_rows = poSrc->GetRasterXSize();  // height
  *init_cols = poSrc->GetRasterYSize();  // width
  *init_bands = poSrc->GetRasterCount();
  GDALClose((GDALDatasetH)poSrc);
  poSrc = NULL;
}

/// \Get the minimum bounding rectangle data of one-specified class.
///
/// \param[in] device_num, the number of device for training (max = 8).
/// \param[in] rank_id, the current device ID.
/// \param[in] image_path, big_input image path.
/// \param[in] label_path, big_input label path.
/// \param[in] n_classes, num classes of labels.
/// \param[in] ignore_label, pad value of ground features.
/// \param[in] seg_threshold, segmentation settings.
/// \param[in] block_size, basic processing unit for big_input data.
/// \param[in] max_searchsize, max output data size (max_searchsize x max_searchsize).
/// \return out, image-label objects in Numpy dtype.
void get_objects(vector<string>& image_path, vector<string>& label_path,
                     int n_classes, int ignore_label, int seg_threshold,
                     int search_block_size, int max_searchsize,
                     string& json_filename, bool use_quadsearch) {

  cout << "Start to get objects." << std::endl;
  
  // Struct for store information of export patch.
  typedef struct {
    vector<int> id;
    pair<vector<string>, vector<string>> data_path;  // <image_path, label_path>
    pair<vector<int>, vector<int>> patch_cord;  // <x, y>
    pair<vector<int>, vector<int>> block_size;  // <block_x, block_y> 
    pair<vector<int>, vector<int>> init_data_cord;  // <width, height>
  } Patch;

  Patch patch;

  int data_id = 0;
  for (int index = 0; index < (int)image_path.size(); index++) {
    // Get init info of big_input.
    int init_cols = 0, init_rows = 0, init_bands = 0;
    get_init_cols_rows(image_path[index], &init_cols, &init_rows, &init_bands);

    // Get info of slice patches.
    for (int i = 0; i < init_rows; i += max_searchsize) {
      for (int j = 0; j < init_cols; j += max_searchsize) {
        int current_block_rows = max_searchsize;
        int current_block_cols = max_searchsize;
        if (i + max_searchsize > init_rows) {
          current_block_rows = init_rows - i;
        }
        if (j + max_searchsize > init_cols) {
          current_block_cols = init_cols - j;
        }

        // Document the patch info.
        patch.id.push_back(data_id);
        patch.data_path.first.push_back(image_path[index]);
        patch.data_path.second.push_back(label_path[index]);
        patch.patch_cord.first.push_back(i);
        patch.patch_cord.second.push_back(j);
        patch.block_size.first.push_back(current_block_rows);
        patch.block_size.second.push_back(current_block_cols);
        patch.init_data_cord.first.push_back(init_rows);
        patch.init_data_cord.second.push_back(init_cols);

        // Get current patch id.
        data_id++;
      }
    }

    if (use_quadsearch) {
      // 1. Sequentially store cord of all-class related global data blocks.
      BlockRead blockread;
      blockread.get_related_block(image_path[index], label_path[index], init_cols, init_rows, n_classes, ignore_label, search_block_size, max_searchsize);
      vector<Vector2> related_block_cord = blockread.get_related_block_cord();

      // 2. In each device, read all class-related data block for quadtree seg and search.
      // (row_cord, col_cord, current_block_rows, current_block_cols)
      for (int i = 0; i < (int)related_block_cord.size(); i++) {
        int row_cord = related_block_cord[i].x;
        int col_cord = related_block_cord[i].y;

        int current_block_rows = search_block_size;
        int current_block_cols = search_block_size;

        // Process the residual data block of big_input data.
        if (row_cord + search_block_size > init_rows) {
          current_block_rows = init_rows - row_cord;
        }
        if (col_cord + search_block_size > init_cols) {
          current_block_cols = init_cols - col_cord;
        }

        // Read original big_input data by basic processing unit, and transfer data to cv::Mat data type.
        GDAL2CV gdal2cv;
        Mat image = gdal2cv.gdal_read(image_path[index], row_cord, col_cord, current_block_rows, current_block_cols);
        Mat label = gdal2cv.gdal_read(label_path[index], row_cord, col_cord, current_block_rows, current_block_cols);

        // Create data pyramid.
        Pyramid pyramid;
        pyramid.create_pyramid(image, label);
        vector<Mat> image_pyramid = pyramid.get_image_pyramid();
        vector<Mat> label_pyramid = pyramid.get_label_pyramid();
        Mat top_level_image = image_pyramid.back();
        Mat top_level_label = label_pyramid.back();
        Mat ori_level_image = image_pyramid.front();
        Mat ori_level_label = label_pyramid.front();

        // Create quadtree and search class-related data block.
        QuadTree quadtree;
        quadtree.random_search(top_level_image, top_level_label, n_classes, ignore_label, seg_threshold);
        quadtree.get_multiscale_object(ori_level_image, ori_level_label, max_searchsize,
          init_rows, init_cols,
          row_cord, col_cord, current_block_rows, current_block_cols);

        // Get search_patch info.
        pair<vector<int>, vector<int>> search_patch_cord = quadtree.get_patch_cord();  // <row_cord, col_cord>
        pair<vector<int>, vector<int>> search_patch_block_size = quadtree.get_block_size();  // <row_block, col_block>

        for (int i = 0; i < (int)search_patch_cord.first.size(); i++) {

          patch.id.push_back(data_id);
          patch.data_path.first.push_back(image_path[index]);
          patch.data_path.second.push_back(label_path[index]);
          patch.patch_cord.first.push_back(search_patch_cord.first[i]);
          patch.patch_cord.second.push_back(search_patch_cord.second[i]);
          patch.block_size.first.push_back(search_patch_block_size.first[i]);
          patch.block_size.second.push_back(search_patch_block_size.second[i]);
          patch.init_data_cord.first.push_back(init_rows);
          patch.init_data_cord.second.push_back(init_cols);

          // Get current patch id.
          data_id++;
        }
      }
    }
  }

  // Patch info -> Json file
  char buff[250];
  //_getcwd(buff, 250);
  getcwd(buff, 250);
  string current_working_directory(buff);
  //cout << current_working_directory << std::endl;
  //string output_json_directory = current_working_directory.append("\\patch_info.json");
  //string output_json_directory = current_working_directory.append("\\" + json_filename);
  //string output_json_directory = current_working_directory.append("/" + json_filename);
  string output_json_directory = current_working_directory.append(json_filename);
  cout << output_json_directory << endl;

  ofstream fout;
  fout.open(output_json_directory, std::ios_base::out | std::ios_base::binary);
  if (!fout.is_open()) {
    std::cout << "open error" << endl;
  }

  int count = 0;
  string document;  // Store Json
  while (count < data_id) {
    Json patch_json2 = Json::object{
      { "imagePath", patch.data_path.first[count] },
      { "labelPath", patch.data_path.second[count] },
      { "x", patch.patch_cord.first[count] },
      { "y", patch.patch_cord.second[count] },
      { "block_x", patch.block_size.first[count] },
      { "block_y", patch.block_size.second[count] },
      { "height", patch.init_data_cord.first[count] },
      { "width", patch.init_data_cord.second[count] },
    };

    Json patch_json = Json::object{
        { to_string(count), patch_json2 },
    };

    string json_obj_str = patch_json.dump();
    document.append(json_obj_str);
    count++;
  }

  // Output string to Json file.
  StringReplacer sr;
  string changed_document = sr.replaceSubstrings(document, "}{", ", ");
  fout << changed_document;
  fout.close();

  cout << "Get objects done." << endl;
}

///// \Get the minimum bounding rectangle data of one-specified class.
/////
///// \param[in] device_num, the number of device for training (max = 8).
///// \param[in] rank_id, the current device ID.
///// \param[in] image_path, big_input image path.
///// \param[in] label_path, big_input label path.
///// \param[in] n_classes, num classes of labels.
///// \param[in] ignore_label, pad value of ground features.
///// \param[in] seg_threshold, segmentation settings.
///// \param[in] block_size, basic processing unit for big_input data.
///// \param[in] max_searchsize, max output data size (max_searchsize x max_searchsize).
///// \return out, image-label objects in Numpy dtype.
//py::list get_objects(int device_num, int rank_id,
//	                   string& image_path, string& label_path,
//	                   int n_classes, int ignore_label, int seg_threshold,
//	                   int block_size, int max_searchsize) {
//
//	// Struct for store export data.
//	typedef struct {
//		vector<Mat> image_objects;
//		vector<Mat> label_objects;
//	} Object;
//	Object object;
//
//	// Get init information of big_input.
//	int init_cols = 0, init_rows = 0, init_bands = 0;
//	get_init_cols_rows(image_path, &init_cols, &init_rows, &init_bands);
//
//	// 1. Sequentially store cord of all-class related data blocks.
//	BlockRead blockread;
//	blockread.get_related_block(image_path, label_path, init_cols, init_rows, n_classes, ignore_label, block_size, max_searchsize);
//	vector<Vector2> related_block_cord = blockread.get_related_block_cord();
//
//  // Get slice patches from original data, put them into std::vector.
//  vector<Mat> image_patches = blockread.get_image_patches();
//  vector<Mat> label_patches = blockread.get_label_patches();
//  object.image_objects = image_patches;
//  object.label_objects = label_patches;
//
//	// 2. In each device, read all class-related data block for quadtree seg and search.
//	for (int i = 0; i < (int) related_block_cord.size(); i++) {
//		int row_cord = related_block_cord[i].x;
//		int col_cord = related_block_cord[i].y;
//
//		int current_block_rows = block_size;
//		int current_block_cols = block_size;
//
//		// Process the residual data block of big_input data.
//		if (row_cord + block_size > init_rows) {
//			current_block_rows = init_rows - row_cord;
//		}
//		if (col_cord + block_size > init_cols) {
//			current_block_cols = init_cols - col_cord;
//		}
//
//		GDAL2CV gdal2cv;
//		if (init_bands > 3) {
//			Mat image = band_selection(image_path, col_cord, row_cord, current_block_cols, current_block_rows);
//		}
//
//		// Read original big_input data by small blocks, and transfer data to cv::Mat data type.
//		Mat image = gdal2cv.gdal_read(image_path, col_cord, row_cord, current_block_cols, current_block_rows);
//		Mat label = gdal2cv.gdal_read(label_path, col_cord, row_cord, current_block_cols, current_block_rows);
//
//		// Create data pyramid.
//		Pyramid pyramid;
//		pyramid.create_pyramid(image, label);
//
//		vector<Mat> image_pyramid = pyramid.get_image_pyramid();
//		vector<Mat> label_pyramid = pyramid.get_label_pyramid();
//
//		Mat top_level_image = image_pyramid.back();
//		Mat top_level_label = label_pyramid.back();
//
//		Mat ori_level_image = image_pyramid.front();
//		Mat ori_level_label = label_pyramid.front();
//
//		// Create quadtree and search class-related data block.
//		QuadTree quadtree;
//		quadtree.random_search(top_level_image, top_level_label, n_classes, ignore_label, seg_threshold);
//		quadtree.get_multiscale_object(ori_level_image, ori_level_label, max_searchsize);
//
//		// Get the output data objects.
//		vector<Mat> image_objects = quadtree.get_image_objects();
//		vector<Mat> label_objects = quadtree.get_label_objects();
//		object.image_objects.insert(object.image_objects.end(), image_objects.begin(), image_objects.end());
//		object.label_objects.insert(object.label_objects.end(), label_objects.begin(), label_objects.end());
//	}
//
//	//// 2. Ditribute data blocks in multiple devices (max = 8).
//	//int num_class_related_block_cord = related_block_cord.size();
//	//int num_one_device_related_block_cord = num_class_related_block_cord / device_num;
//	//int num_residual_related_block_cord = num_class_related_block_cord % device_num;
//	//int sequence_beg_index = 0;
//	//if (rank_id == device_num - 1) {
//	//	if (num_residual_related_block_cord > 0) {
//	//		sequence_beg_index = rank_id * num_one_device_related_block_cord;
//	//		num_one_device_related_block_cord = num_class_related_block_cord - (rank_id * num_one_device_related_block_cord);
//	//	}
//	//}
//	//else {
//	//	sequence_beg_index = rank_id * num_one_device_related_block_cord;
//	//}
//
//	//// 3. Read class-related data block for quadtree seg and search.
//	//for (int i = 0; i < num_one_device_related_block_cord; i++) {
//	//	// The data blocks is depend on device rank_id.
//	//	int row_cord = related_block_cord[sequence_beg_index + i].x;
//	//	int col_cord = related_block_cord[sequence_beg_index + i].y;
//
//	//	int current_block_rows = block_size;
//	//	int current_block_cols = block_size;
//
//	//	// Process the residual data block of big_input data.
//	//	if (row_cord + block_size > init_rows) {
//	//		current_block_rows = init_rows - row_cord;
//	//	}
//	//	if (col_cord + block_size > init_cols) {
//	//		current_block_cols = init_cols - col_cord;
//	//	}
//
//	//	GDAL2CV gdal2cv;
//	//	if (init_bands > 3) {
//	//		Mat image = band_selection(image_path, col_cord, row_cord, current_block_cols, current_block_rows);
//	//	}
//
//	//	// Read original big_input data by small blocks, and transfre data to cv::Mat data type.
//	//	Mat image = gdal2cv.gdal_read(image_path, col_cord, row_cord, current_block_cols, current_block_rows);
//	//	Mat label = gdal2cv.gdal_read(label_path, col_cord, row_cord, current_block_cols, current_block_rows);
//
//	//	// Create data pyramid.
//	//	Pyramid pyramid;
//	//	pyramid.create_pyramid(image, label);
//
//	//	vector<Mat> image_pyramid = pyramid.get_image_pyramid();
//	//	vector<Mat> label_pyramid = pyramid.get_label_pyramid();
//
//	//	Mat top_level_image = image_pyramid.back();
//	//	Mat top_level_label = label_pyramid.back();
//
//	//	Mat ori_level_image = image_pyramid.front();
//	//	Mat ori_level_label = label_pyramid.front();
//
//	//	// Create quadtree and search class-related data block.
//	//	QuadTree quadtree;
//	//	quadtree.random_search(top_level_image, top_level_label, n_classes, ignore_label, seg_threshold);
//	//	quadtree.get_multiscale_object(ori_level_image, ori_level_label, max_searchsize);
//
//	//	// Get the output data objects.
//	//	vector<Mat> image_objects = quadtree.get_image_objects();
//	//	vector<Mat> label_objects = quadtree.get_label_objects();
//	//	object.image_objects.insert(object.image_objects.end(), image_objects.begin(), image_objects.end());
//	//	object.label_objects.insert(object.label_objects.end(), label_objects.begin(), label_objects.end());
//	//}
//
//	// If the output data objects is empty, read data in traditional method.
//	// TODO: To deal with empty image_objects.
//	if (object.image_objects.empty()) {
//		cout << "Something went wrong in quadtree search, read data blcok by SlicePatches API.";
//	}
//
//  // 3. Ditribute data blocks in multiple devices (max = 8).
//  int num_object = object.image_objects.size();
//  int num_one_device_object = num_object / device_num;
//
//  // For multi-cards training, residual object number in last device
//  // should be the same as other decives (num_one_device_object).
//  int sequence_beg_index = rank_id * num_one_device_object;
//
//  //// 3. Ditribute data blocks in multiple devices (max = 8).
//  //int num_object = object.image_objects.size();
//  //int num_one_device_object = num_object / device_num;
//  //int num_residual_object = num_object % device_num;
//  //int sequence_beg_index = 0;
//
//  //// For multi-cards training, residual object number in last device
//  //// should be the same as other decives (num_one_device_object).
//  //if (rank_id == device_num - 1) {
//  //  //if (num_residual_object > 0 && num_residual_object < num_one_device_object) {
//  //  //  // Copy data to fill, so as to have the same object number as other decives.
//  //  //  sequence_beg_index = rank_id * num_one_device_object;
//  //  //  Mat copy2fill_image = object.image_objects[sequence_beg_index + num_residual_object - 1];
//  //  //  Mat copy2fill_label = object.label_objects[sequence_beg_index + num_residual_object - 1];
//
//  //  //  object.image_objects.insert(object.image_objects.end(), (num_one_device_object-num_residual_object), copy2fill_image);
//  //  //  object.label_objects.insert(object.label_objects.end(), (num_one_device_object - num_residual_object), copy2fill_label);
//  //  //}
//
//  //  // If num_residual_object > 0, num_residual_object must be larger than num_one_device_object.
//  //  if (num_residual_object > 0) {
//  //    // Ignore the surplus data, so as to have the same object number as other decives.
//  //    object.image_objects.erase(object.image_objects.begin() + (device_num * num_one_device_object),
//  //                               object.image_objects.begin() + num_object);
//  //    object.label_objects.erase(object.label_objects.begin() + (device_num * num_one_device_object),
//  //                               object.label_objects.begin() + num_object);
//  //    sequence_beg_index = rank_id * num_one_device_object;
//  //  }
//  //}
//  //else {
//  //	sequence_beg_index = rank_id * num_one_device_object;
//  //}
//
//	//// 3. Ditribute data blocks in multiple devices (max = 8).
//	//int num_object = object.image_objects.size();
//	//int num_one_device_object = num_object / device_num;
//	//int num_residual_object = num_object % device_num;
//	//int sequence_beg_index = 0;
//	//if (rank_id == device_num - 1) {
//	//	if (num_residual_object > 0) {
//	//		sequence_beg_index = rank_id * num_one_device_object;
//	//		num_one_device_object = num_object - (rank_id * num_one_device_object);
//	//	}
//	//}
//	//else {
//	//	sequence_beg_index = rank_id * num_one_device_object;
//	//}
//
//	// 4. For output data objects, convert cv::Mat data type to numpy data type.
//	py::list out_image_objects, out_label_objects;
//	for (int index = 0; index < num_one_device_object; index++) {
//		// image objects.
//		Mat src_image = object.image_objects[sequence_beg_index + index];
//		py::array_t<unsigned char> dst_image = cv_mat_uint8_3c_to_numpy(src_image);
//		out_image_objects.append(dst_image);
//
//		// label objects.
//		Mat src_label = object.label_objects[sequence_beg_index + index];
//		py::array_t<unsigned char> dst_label = cv_mat_uint8_1c_to_numpy(src_label);
//		out_label_objects.append(dst_label);
//	}
//
//	py::list out;
//	out.append(out_image_objects);
//	out.append(out_label_objects);
//	return out;
//
//	//// 4. For output data objects, convert cv::Mat data type to numpy data type.
//	//py::list out_image_objects, out_label_objects;
//	//for (int index = 0; index < (int) object.image_objects.size(); index++) {
//	//	// image objects.
//	//	Mat src_image = object.image_objects[index];
//	//	py::array_t<unsigned char> dst_image = cv_mat_uint8_3c_to_numpy(src_image);
//	//	out_image_objects.append(dst_image);
//
//	//	// label objects.
//	//	Mat src_label = object.label_objects[index];
//	//	py::array_t<unsigned char> dst_label = cv_mat_uint8_1c_to_numpy(src_label);
//	//	out_label_objects.append(dst_label);
//	//}
//
//	//py::list out;
//	//out.append(out_image_objects);
//	//out.append(out_label_objects);
//	//return out;
//}

//py::list get_objects(const int device_num, const int rank_id,
//	                   const string& image_path, const string& label_path,
//	                   const int n_classes, const int ignore_label=255,
//	                   const int block_size=4096, const int max_searchsize=2048) {
//	// Struct for store export data.
//	typedef struct {
//		vector<Mat> image_objects;
//		vector<Mat> label_objects;
//	} Object;
//	Object object;
//
//	// Get init information of big_input.
//	int init_cols, init_rows, init_bands = 0;
//	get_init_cols_rows(image_path, &init_cols, &init_rows, &init_bands);
//
//	Mat image, label;
//	if (init_cols <= block_size && init_rows <= block_size) {
//		// Todo: The support for high-spectral data (more than 3 bands).
//		if (init_bands > 3) {
//			image = band_selection(image_path, 0, 0, init_cols, init_rows);
//		}
//		else {
//			image = cv::imread(image_path, -1);
//		}
//		label = cv::imread(label_path, -1);
//
//		// Create data pyramid.
//		Pyramid pyramid;
//		pyramid.create_pyramid(image, label);
//		Mat top_level_image = pyramid.get_image_pyramid.back();
//		Mat top_level_label = pyramid.get_label_pyramid.back();
//		Mat ori_level_image = pyramid.get_image_pyramid.front();
//		Mat ori_level_label = pyramid.get_label_pyramid.front();
//
//		// Create quadtree search tree.
//		QuadTree quadtree;
//		quadtree.random_search(top_level_image, top_level_label, n_classes, ignore_label);
//		quadtree.get_multiscale_object(ori_level_image, ori_level_label, max_searchsize);
//		// Get the output data objects.
//		object.image_objects = quadtree.get_image_objects;
//		object.label_objects = quadtree.get_label_objects;
//	}
//	else {
//		// The cord of all-class related data blocks are sequentially stored in related_block_cord.
//		BlockRead blockread;
//		blockread.get_related_block(label_path, init_cols, init_rows, n_classes, ignore_label, block_size);
//		vector<Vector2> related_block_cord = blockread.get_related_block_cord();
//
//		// The data blocks are sequentially processed depend on device num and rank_id of current device.
//		int num_class_related_block_cord = related_block_cord.size();
//		int num_one_device_related_block_cord = num_class_related_block_cord / device_num;
//		int num_residual_related_block_cord = num_class_related_block_cord % device_num;
//		int sequence_beg_index = 0;
//
//		// The last device is responsible to process residual data blocks.
//		if (num_residual_related_block_cord > 0) {
//			if (rank_id == device_num - 1) {
//				sequence_beg_index = rank_id * num_one_device_related_block_cord;
//				num_one_device_related_block_cord = 
//					num_class_related_block_cord - (rank_id * num_one_device_related_block_cord);
//			}
//		}
//		else {
//			sequence_beg_index = rank_id * num_one_device_related_block_cord;
//		}
//		// for each class-related data block, read their original cord in an sequence.
//		for (int i = 0; i < num_one_device_related_block_cord; i++) {
//			int row_cord = related_block_cord[sequence_beg_index + i].x;
//			int col_cord = related_block_cord[sequence_beg_index + i].y;
//			int current_block_rows = block_size;
//			int current_block_cols = block_size;
//
//			// Process the residual data block of big_input data.
//			if (row_cord + block_size > init_rows) {
//				current_block_rows = init_rows - row_cord;
//			}
//			if (col_cord + block_size > init_cols) {
//				current_block_cols = init_cols - col_cord;
//			}
//			GDAL2CV gdal2cv;
//			if (init_bands > 3) {
//				image = band_selection(image_path, col_cord, row_cord, current_block_cols, current_block_rows);
//			}
//			else {
//				image = gdal2cv.gdal_read(image_path, col_cord, row_cord, current_block_cols, current_block_rows);
//			}
//			label = gdal2cv.gdal_read(label_path, col_cord, row_cord, current_block_cols, current_block_rows);
//
//			// Create data pyramid.
//			Pyramid pyramid;
//			pyramid.create_pyramid(image, label);
//			Mat top_level_image = pyramid.get_image_pyramid.back();
//			Mat top_level_label = pyramid.get_label_pyramid.back();
//			Mat ori_level_image = pyramid.get_image_pyramid.front();
//			Mat ori_level_label = pyramid.get_label_pyramid.front();
//
//			// Create quadtree search tree.
//			QuadTree quadtree;
//			quadtree.random_search(top_level_image, top_level_label, n_classes, ignore_label);
//			quadtree.get_multiscale_object(ori_level_image, ori_level_label, max_searchsize);
//
//			// Get the output data objects.
//			object.image_objects.insert(object.image_objects.end(),
//																  quadtree.get_image_objects.begin(), quadtree.get_image_objects.end());
//			object.label_objects.insert(object.label_objects.end(),
//																	quadtree.get_label_objects.begin(), quadtree.get_label_objects.end());
//		}
//	}
//	// If the output data objects is empty, throw error information.
//	// TODO: how to deal with empty image_objects.
//	if (object.image_objects.empty()) {
//		cout << "Something went wrong in quadtree search, read blcok in origin method.";
//		//GDAL2CV gdal2cv;
//		//Mat image = gdal2cv.gdal_read(image_path, 0, 0, 2048, 2048);
//		//Mat label = gdal2cv.gdal_read(label_path, 0, 0, 2048, 2048);
//		//object.image_objects.push_back(image);
//		//object.label_objects.push_back(label);
//	}
//
//	// Convert the Mat dtype to Numpy dtype.
//	Mat src_image, src_label;
//	py::array_t<unsigned char> dst_image, dst_label;
//	py::list out_image_objects, out_label_objects;
//	for (int index = 0; index < (int) object.image_objects.size(); index++) {
//		// image objects.
//		src_image = object.image_objects[index];
//		dst_image = cv_mat_uint8_3c_to_numpy(src_image);
//		out_image_objects.append(dst_image);
//
//		// label objects.
//		src_label = object.label_objects[index];
//		dst_label = cv_mat_uint8_1c_to_numpy(src_label);
//		out_label_objects.append(dst_label);
//	}
//	py::list out;
//	out.append(out_image_objects);
//	out.append(out_label_objects);
//	return out;
//}

// Python API.
PYBIND11_MODULE(geobject, m) {
	m.doc() = "input filename, output py::list for objects";
	// Add bindings function.
	m.def("get_objects", &get_objects, "A function which gets the ground objects in big_input data");
}

}	// namespace luojianet_ms
