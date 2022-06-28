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

#ifndef QUADTREE_H_
#define QUADTREE_H_

#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Vector2.h"
#include "boundbox.h"
#include "quadnode.h"

using std::cout;
using std::queue;
using std::random_device;
using std::mt19937;
using std::vector;
using std::pair;
using cv::Mat;
using cv::Mat_;
using cv::cvtColor;
using cv::COLOR_BGR2GRAY;
using cv::Range;
using cv::minMaxLoc;
using cv::countNonZero;

namespace luojianet_ms {

class QuadTree {
 public:
	QuadTree();
	~QuadTree();

	/// \Search label-value realted areas from init quadtree nodes.
	/// \param[in] top_level_image, top level pyramid image.
	/// \param[in] top_level_label, top level pyramid label.
	/// \param[in] n_classes, num classes of labels.
	/// \param[in] ignore_label, the value to skip over in statistic.
	void random_search(Mat& top_level_image, Mat& top_level_label, int n_classes, int ignore_label, int seg_threshold);

	/// \Create quadtree nodes based on top level pyramid data.
	/// \param[in] top_level_image, top level pyramid image.
	/// \param[in] top_level_label, top level pyramid label.
	/// \param[in] n_classes, num classes of labels.
	/// \param[in] ignore_label, the value to skip over in statistic.
	void create_quadtree(Mat& top_level_image, Mat& top_level_label, int n_classes, int ignore_label, int seg_threshold);

	/// \Decide to subdivide or not.
	/// \param[in] gray_image, only for 1-channel image data.
	/// \param[in] root_node, init is the root node, after quadtree subdivide, root_node has all nodes for this area.
	bool sub_or_not(Mat& image, QuadNode* root_node, int seg_threshold);

	/// \Label all quadtree subdivide nodes for this area.
	/// \param[in] top_level_label, top level pyramid label.
	/// \param[in] n_classes, num classes of labels.
	/// \param[in] ignore_label, the value to skip over in statistic.
	/// \param[in] root_node, init is the root node, after quadtree subdivide, root_node has all nodes for this area.
	void label_quadtree(Mat& top_level_label, int n_classes, int ignore_label, QuadNode* root_node);

	/// \Store all quadtree nodes in 'nodelist' for grid_search.
	/// \param[in] root_node, init is the root node, after quadtree subdivide, root_node has all nodes for this area.
	void store_node(QuadNode* root_node);

	/// \For each quadtree node, use 8-neighborhood grid search method,
	/// \to find grids which has the same main label-value.
	/// \param[in] top_level_label, top level pyramid label.
	/// \param[in] n_classes, num classes of labels.
	/// \param[in] ignore_label, the value to skip over in statistic.
	void grid_search(Mat& top_level_label, int n_classes, int ignore_label);

	/// \Mapping the cord of grid search results in top_level_data to ori_level_data.
	/// \In order to accelerate processing, directly mapping cord to original size.
	/// \param[in] ori_level_image, original level pyramid image.
	/// \param[in] ori_level_label, original level pyramid label.
	/// \param[in] max_searchsize, the maximum search size in origin cord.
	void get_multiscale_object(Mat& ori_level_image, Mat& ori_level_label, int max_searchsize);

	/// \Release the memory.
	/// \param[in] root_node, init is the root node, after quadtree subdivide, root_node has all nodes for this area.
	void delete_quadtree(QuadNode* root_node);

	vector<Mat> get_image_objects() const { return image_objects; }
	vector<Mat> get_label_objects() const { return label_objects; }

	//void quad_search();

	///// \Store all quadtree nodes in nodelist for random select and search.
	///// \param[in] root_node, init is the root node, after quadtree subdivide, root_node has all nodes for this area.
	//void store_node(QuadNode* root_node);

	///// \Use Preorder traversal search in quadtree structure,
	///// \to find nodes which has the same main label-value.
	///// \param[in] root_node, init is the root node, after quadtree subdivide, root_node has all nodes for this area.
	///// \param[in] random_node, init search node.
	///// \param[in] nodepath, the searching path.
	///// \param[in] temp_nodepath, store the temporary node path.
	///// \param[in] flag, if find the same main label-value node, flag = 1, otherwise flag = 0.
	//void get_nodepath(QuadNode* root_node, QuadNode* random_node,
	//									vector<QuadNode*>& nodepath, vector<QuadNode*>& temp_nodepath, int flag=0);

 private:
	//const int kSegThreshold = 150;  // segmentation settings.
	//const int kNodes = 64;  // quatree settings.

	QuadNode* root_node;  // nodes for quadtree.
	vector<QuadNode*> nodelist;  // store all quadtree nodes for grid search.
	//vector<QuadNode*> allnode_quadsearch_result;

	vector<Mat_<uchar>> top_level_mask;  // masks for grid search results.

	vector<Mat> image_objects;  // random_search image results.
	vector<Mat> label_objects;  // random_search label results.
};

}  // namespace luojianet_ms

#endif  // QUADTREE_H_