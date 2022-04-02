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
#include <gdal_priv.h>
#include <gdal.h>
#include <opencv2/opencv.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "blockread.h"
#include "pyramid.h"
#include "quadtree.h"
#include "gdal2cv.h"

namespace py = pybind11;
using namespace std;


void get_init_cols_rows(string &image_path, int *init_cols, int *init_rows, int *init_bands) {
	const char *imagepath = image_path.data();
	GDALAllRegister();
	GDALDataset *poSrc = (GDALDataset*)GDALOpen(imagepath, GA_ReadOnly);
	if (poSrc == NULL) {
		cout << "GDAL failed to open " << imagepath;
		return;
	}
	*init_cols = poSrc->GetRasterXSize();
	*init_rows = poSrc->GetRasterYSize();
	*init_bands = poSrc->GetRasterCount();

	GDALClose((GDALDatasetH)poSrc);
	poSrc = NULL;
}


// One channel data, Mat->Numpy
py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(cv::Mat &input) {
	py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows,input.cols }, input.data);
	return dst;
}

// Three channel data, Mat->Numpy
py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(cv::Mat &input) {
	py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows, input.cols, 3 }, input.data);
	return dst;
}


py::list get_objects(int device_num, int rank_id,
                    string &image_path, string &label_path,
                    int n_classes, int ignore_label,
                    int block_size, int max_searchsize, int min_searchsize) {
	typedef struct {
		vector<cv::Mat> image_objects;
		vector<cv::Mat> label_objects;
	} Object;
	Object object;

	int init_cols, init_rows, init_bands = 0;
	get_init_cols_rows(image_path, &init_cols, &init_rows, &init_bands);

	cv::Mat image, label;
    // Todo: The support for high-spectral data
	if (init_cols <= block_size && init_rows <= block_size) {
		if (init_bands > 3) {
			cout << "NotImplementedError" << endl;
            throw runtime_error("Unknown image depth, gdal: 1, img: 1");
		}
		else {
			image = cv::imread(image_path, -1);
		}
		label = cv::imread(label_path, -1);

		Pyramid pyramid;
		pyramid.create_pyramid(image, label);
		cv::Mat top_level_image = pyramid.image_pyramid.back();
		cv::Mat top_level_label = pyramid.label_pyramid.back();
		cv::Mat ori_level_image = pyramid.image_pyramid.front();
		cv::Mat ori_level_label = pyramid.label_pyramid.front();

		QuadTree quadtree;
		quadtree.random_search(top_level_image, top_level_label, n_classes, ignore_label);
		quadtree.get_multiscale_object(ori_level_image, ori_level_label, max_searchsize, min_searchsize);

		object.image_objects = quadtree.image_objects;
		object.label_objects = quadtree.label_objects;
	}
	else {
		BlockRead blockread;
		blockread.get_related_block(label_path, init_cols, init_rows, n_classes, ignore_label, block_size);
		vector<Vector2> related_block_cord = blockread.related_block_cord;

        // The cord of all-class related blocks are sequentially stored in vector<Vector2>related_block_cord.
		int num_class_related_block_cord = related_block_cord.size();

        // The blocks are sequentially processed depend on device num and rank_id.
        int num_one_device_related_block_cord = num_class_related_block_cord / device_num;
        int residual_related_block_cord = num_class_related_block_cord % device_num;
        int sequence_beg_index = 0;
        if (residual_related_block_cord > 0) {
            // The last device is responsible to process residual blocks.
            if (rank_id == device_num - 1) {
                sequence_beg_index = rank_id * num_one_device_related_block_cord;
                num_one_device_related_block_cord = num_class_related_block_cord - (rank_id * num_one_device_related_block_cord);
            }
        }
        else {
            sequence_beg_index = rank_id * num_one_device_related_block_cord;
        }
		// for each class realted block, read their orignal cord in an sequence.
		for (int i = 0; i < num_one_device_related_block_cord; i++) {
			int row_cord = related_block_cord[sequence_beg_index+i].x;
			int col_cord = related_block_cord[sequence_beg_index+i].y;
			int current_block_rows = block_size;
			int current_block_cols = block_size;
			if (row_cord + block_size > init_rows) {
				current_block_rows = init_rows - row_cord;
			}
			if (col_cord + block_size > init_cols) {
				current_block_cols = init_cols - col_cord;
			}

			GDAL2CV gdal2cv;
			// For high-spectral images, just select the first three bands.
            // Todo: The support for auto select informative bands.
			if (init_bands > 3) {
				image = gdal2cv.gdal_read(image_path, col_cord, row_cord, current_block_cols, current_block_rows);
				vector<cv::Mat> all_channels;
				vector<cv::Mat> select_channels;
				split(image, all_channels);
				for (int i = 0; i < 3; i++) {
					select_channels.push_back(all_channels.at(i));
				}
				merge(select_channels, image);
			}
			else {
				image = gdal2cv.gdal_read(image_path, col_cord, row_cord, current_block_cols, current_block_rows);
			}
			label = gdal2cv.gdal_read(label_path, col_cord, row_cord, current_block_cols, current_block_rows);

			Pyramid pyramid;
			pyramid.create_pyramid(image, label);
			cv::Mat top_level_image = pyramid.image_pyramid.back();
			cv::Mat top_level_label = pyramid.label_pyramid.back();
			cv::Mat ori_level_image = pyramid.image_pyramid.front();
			cv::Mat ori_level_label = pyramid.label_pyramid.front();

			QuadTree quadtree;
			quadtree.random_search(top_level_image, top_level_label, n_classes, ignore_label);
			quadtree.get_multiscale_object(ori_level_image, ori_level_label, max_searchsize, min_searchsize);

			object.image_objects.insert(object.image_objects.end(), quadtree.image_objects.begin(), quadtree.image_objects.end());
			object.label_objects.insert(object.label_objects.end(), quadtree.label_objects.begin(), quadtree.label_objects.end());
		}
	}

    // If the vector<cv::Mat> image_objects is empty.
	if (object.image_objects.empty()) {
		GDAL2CV gdal2cv;
		cv::Mat image, label;
		image = gdal2cv.gdal_read(image_path, 0, 0, 1024, 1024);
		label = gdal2cv.gdal_read(label_path, 0, 0, 1024, 1024);
		object.image_objects.push_back(image);
		object.label_objects.push_back(label);
	}

	cv::Mat src_image_objects, src_label_objects;
    py::array_t<unsigned char> dst_image_objects, dst_label_objects;
	py::list out_image_objects, out_label_objects;

	int size = object.image_objects.size();
    for (int index = 0; index < size; index++) {
        src_image_objects = object.image_objects[index];
        dst_image_objects = cv_mat_uint8_3c_to_numpy(src_image_objects);
        out_image_objects.append(dst_image_objects);

        src_label_objects = object.label_objects[index];
        dst_label_objects = cv_mat_uint8_1c_to_numpy(src_label_objects);
        out_label_objects.append(dst_label_objects);
    }

    py::list out;
    out.append(out_image_objects);
    out.append(out_label_objects);

	return out;
}


PYBIND11_MODULE(geobject, m) {

	m.doc() = "input filename, output py::list for objects";

	// Add bindings here
	m.def("get_objects", &get_objects);
}