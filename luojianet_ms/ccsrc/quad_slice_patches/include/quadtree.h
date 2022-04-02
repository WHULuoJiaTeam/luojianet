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

#ifndef QUADTREE_H
#define QUADTREE_H

#include "Vector2.h"
#include "boundbox.h"
#include "quadnode.h"

#include <random>

#include <opencv2/opencv.hpp>

using namespace std;

// Segmentation settings
#define SEG_THRESHOLD 25

// Quatree settings
#define N_NODES 64


class QuadTree {
public:
	QuadNode* root_node;
	vector<QuadNode*> nodelist;
	vector<QuadNode*> allnode_quadsearch_result;

	vector<cv::Mat_<uchar>> top_level_mask;
	vector<cv::Mat> image_objects;
	vector<cv::Mat> label_objects;

public:
	QuadTree();
	~QuadTree();

	void random_search(cv::Mat &top_level_image, cv::Mat &top_level_label, int n_classes, int ignore_label);

	void create_quadtree(cv::Mat &top_level_image, cv::Mat &top_level_label, int n_classes, int ignore_label);
	void label_quadtree(cv::Mat &top_level_label, int n_classes,  int ignore_label, QuadNode* root_node);
	bool sub_or_not(cv::Mat &image, QuadNode* root_node);

	void quad_search();
	void store_node(QuadNode* root_node);
	void get_nodepath(QuadNode* root_node, QuadNode* random_node, vector<QuadNode*> &nodepath, vector<QuadNode*> &temp_nodepath, int flag);

	void grid_search(cv::Mat &top_level_label, int n_classes, int ignore_label);

	void get_multiscale_object(cv::Mat &ori_level_image, cv::Mat &ori_level_label, int max_searchsize, int min_searchsize);

	void delete_quadtree(QuadNode* root_node);
};


#endif