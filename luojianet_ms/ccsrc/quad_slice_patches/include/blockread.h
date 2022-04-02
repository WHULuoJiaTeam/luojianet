/*
* Created by: ZhangZhan
* Wuhan University
* zhangzhanstep@whu.edu.cn
* Copyright (c) 2021
*/

#ifndef BLOCKREAD_H
#define BLOCKREAD_H

#include <string>

#include "gdal2cv.h"
#include "pyramid.h"
#include "boundbox.h"

using namespace std;

// Block settings
//#define BLOCK_SIZE 8192


class BlockRead {
public:
	cv::Mat_<uchar> class_attribute;
	vector<cv::Mat_<uchar>> related_class_mask;
	vector<Vector2> related_block_cord;
public:
	BlockRead();
	~BlockRead();

	void get_related_block(string &label_path, int init_cols, int init_rows, int n_classes, int ignore_label, int block_size);

	void get_class_attribute(string &label_path, int init_rows, int init_cols, int n_classes, int ignore_label, int block_size);
	void quick_statistic_class(cv::Mat &label, int block_index, int n_classes, int ignore_label);
	void search_related_block(int init_cols, int init_rows, int block_size);
	void store_related_blockcord(int block_size);
};

#endif