/*
* Created by: ZhangZhan
* Wuhan University
* zhangzhanstep@whu.edu.cn
* Copyright (c) 2021
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