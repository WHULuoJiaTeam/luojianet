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

#include "quadtree.h"

namespace luojianet_ms {

QuadTree::QuadTree() {}

QuadTree::~QuadTree() {}

void QuadTree::random_search(Mat& top_level_image, Mat& top_level_label, int n_classes, int ignore_label, int seg_threshold) {
	create_quadtree(top_level_image, top_level_label, n_classes, ignore_label, seg_threshold);
	//quad_search();
	grid_search(top_level_label, n_classes, ignore_label);
	delete_quadtree(root_node);
}

// Function used in 'create_quadtree'.
Mat rgb2gray(Mat& image) {
	cvtColor(image, image, COLOR_BGR2GRAY);
	return image;
}

// Function used in 'create_quadtree'.
bool QuadTree::sub_or_not(Mat& gray_image, QuadNode* root_node, int seg_threshold) {
	Mat node_image = gray_image(Range(root_node->node_cord.LBx(), root_node->node_cord.UBx()),
															Range(root_node->node_cord.LBy(), root_node->node_cord.UBy()));
	double max_value, min_value;
	minMaxLoc(node_image, &min_value, &max_value);
	int diff = max_value - min_value;
	if (diff < seg_threshold) {
		return false;
	}
	return true;
}

// Function used in 'create_quadtree'.
void QuadTree::label_quadtree(Mat& top_level_label, int n_classes, int ignore_label, QuadNode* root_node) {
	if (root_node == NULL) {
		return;
	}
	Mat node_label = top_level_label(Range(root_node->node_cord.LBx(), root_node->node_cord.UBx()),
										               Range(root_node->node_cord.LBy(), root_node->node_cord.UBy()));
	vector<int> init(n_classes, 0);
	root_node->label_value = init;
	for (int i = 0; i < node_label.rows; i++) {
		for (int j = 0; j < node_label.cols; j++) {
			int value = node_label.at<uchar>(i, j);
			if (value == ignore_label) {
				continue;
			}
			root_node->label_value[value]++;
		}
	}
	label_quadtree(top_level_label, n_classes, ignore_label, root_node->children[0]);
	label_quadtree(top_level_label, n_classes, ignore_label, root_node->children[1]);
	label_quadtree(top_level_label, n_classes, ignore_label, root_node->children[2]);
	label_quadtree(top_level_label, n_classes, ignore_label, root_node->children[3]);
}

void QuadTree::create_quadtree(Mat& top_level_image, Mat& top_level_label, int n_classes, int ignore_label, int seg_threshold) {
	if (!(root_node = new QuadNode())) {
		cout << "Memory Allocate Error.";
		return;
	}

	Mat gray_image = rgb2gray(top_level_image);
	BoundaryBox image_cord(Vector2(0, 0), Vector2(gray_image.rows, gray_image.cols));
	root_node->node_cord = image_cord;
	QuadNode* root_record = root_node;

	queue<QuadNode*> tree_queue;
	tree_queue.push(root_node);
	while (!tree_queue.empty()) {
		root_node = tree_queue.front();
		tree_queue.pop();
		if (root_node->node_cord.GetDims() != 2) {
			if (sub_or_not(gray_image, root_node, seg_threshold)) {
				root_node->sub_quadtree();
				tree_queue.push(root_node->children[0]);
				tree_queue.push(root_node->children[1]);
				tree_queue.push(root_node->children[2]);
				tree_queue.push(root_node->children[3]);
			}
		}
	}
	root_node = root_record;

	// For each node area, count the number of label value.
	label_quadtree(top_level_label, n_classes, ignore_label, root_node);
}

// Function used in 'grid_search'.
void QuadTree::store_node(QuadNode* root_node) {
	if (root_node == NULL) {
		return;
	}
	if (!(root_node->has_children)) {
		nodelist.push_back(root_node);
	}
	store_node(root_node->children[0]);
	store_node(root_node->children[1]);
	store_node(root_node->children[2]);
	store_node(root_node->children[3]);
}

// Funtion used in 'grid_search'.
// Get main label value of the current node area.
int get_node_mainlabel(QuadNode* node) {
	if (node == NULL) {
		cout << "Empty root node.";
		return -1;
	}
	vector<int>::iterator max = max_element(begin(node->label_value), end(node->label_value));
	return distance(begin(node->label_value), max);
}

// Function used in 'grid_search'.
// Get main label value of the current grid area.
int get_grid_mainlabel(Mat& grid_label, int n_classess, int ignore_label) {
	vector<int> grid_label_value(n_classess, 0);
	for (int i = 0; i < grid_label.cols; i++) {
		for (int j = 0; j < grid_label.rows; j++) {
			int value = grid_label.at<uchar>(i, j);
			if (value == ignore_label) {
				continue;
			}
			grid_label_value[value]++;
		}
	}
	vector<int>::iterator max = max_element(begin(grid_label_value), end(grid_label_value));
	return distance(begin(grid_label_value), max);
}

// Function used in 'grid_search'.
void push(vector<BoundaryBox> &temp_grid_result, BoundaryBox &grid) {
	temp_grid_result.push_back(grid);
}

// Function used in 'grid_search'.
bool pop(vector<BoundaryBox> &temp_grid_result, BoundaryBox &grid) {
	if (temp_grid_result.size() < 1) {
		return false;
	}
	grid = temp_grid_result.back();
	temp_grid_result.pop_back();
	return true;
}

// Function used in 'grid_search'.
void QuadTree::grid_search(Mat& top_level_label, int n_classes, int ignore_label) {
	store_node(root_node);
	for (vector<QuadNode*>::iterator it = nodelist.begin(); it != nodelist.end(); it++) {
		Mat_<uchar> istaken(16, 16, uchar(0));  // create cv::Mat to store each grid search results.

		int init_searchlabel = get_node_mainlabel(*it);  // Had skipped over ignore_label.

		int size = (*it)->node_cord.GetSize();
		if (size == 16 * 16) {
			// Ignore the root node of quadtree (whole area).
			continue;
		}

		// Begin 8-neighborhood grid search.
		const Vector2 dxy[8] = {
				Vector2(-size, 0),
				Vector2(-size, -size),
				Vector2(0, -size),
				Vector2(size, -size),
				Vector2(size, 0),
				Vector2(size, size),
				Vector2(0, size),
				Vector2(-size, size) };
		vector<BoundaryBox> temp_grid_result;
		BoundaryBox grid = (*it)->node_cord;
		temp_grid_result.push_back(grid);
		while (pop(temp_grid_result, grid)) {
			istaken(Range(grid.LBx(), grid.UBx()), Range(grid.LBy(), grid.UBy())) = true;
			for (int i = 0; i < 8; i++) {
				Vector2 lowerbound = grid.LowerBound + dxy[i];
				Vector2 upperbound = grid.UpperBound + dxy[i];
				BoundaryBox grid(lowerbound, upperbound);
				// Search in 16 x 16 label area.
				if ((grid.LBx() >= 0 && grid.LBy() >= 0 && grid.LBx() < 16 && grid.LBy() < 16) &&
						(grid.UBx() >= 0 && grid.UBy() >= 0 && grid.UBx() < 16 && grid.UBy() < 16)) {
					if (istaken(Range(grid.LBx(), grid.UBx()), Range(grid.LBy(), grid.UBy())).at<uchar>(0, 0) == false) {
						Mat grid_label = top_level_label(Range(grid.LBx(), grid.UBx()), Range(grid.LBy(), grid.UBy()));
						int grid_mainlabel = get_grid_mainlabel(grid_label, n_classes, ignore_label);
						if (grid_mainlabel == init_searchlabel) {
							push(temp_grid_result, grid);
						}
					}
				}
			}
		}
		top_level_mask.push_back(istaken);
	}
}

// Function used in 'get_multiscale_object'.
// Get the minimum rectangle of all-realted grid area.
BoundaryBox get_mini_rect(Mat& top_level_mask) {
	vector<int> all_true_x;
	vector<int> all_true_y;
	for (int i = 0; i < top_level_mask.rows; i++) {
		for (int j = 0; j < top_level_mask.cols; j++) {
			if (top_level_mask.at<uchar>(i, j) == true) {
				all_true_x.push_back(i);
				all_true_y.push_back(j);
			}
		}
	}

	if (all_true_x.empty()) {
		return BoundaryBox(Vector2(0, 0), Vector2(16, 16));
	}

	// Get max x and min x of minimum rectangle.
	vector<int>::iterator max_x = max_element(begin(all_true_x), end(all_true_x));
	vector<int>::iterator min_x = min_element(begin(all_true_x), end(all_true_x));
	int max_true_x = *max_x;
	int min_true_x = *min_x;

	// Get max y and min y of minimum rectangle.
	vector<int>::iterator max_y = max_element(begin(all_true_y), end(all_true_y));
	vector<int>::iterator min_y = min_element(begin(all_true_y), end(all_true_y));
	int max_true_y = *max_y;
	int min_true_y = *min_y;

	return BoundaryBox(Vector2(min_true_x, min_true_y), Vector2(max_true_x, max_true_y));
}

void QuadTree::get_multiscale_object(Mat& ori_level_image, Mat& ori_level_label, int max_searchsize,
                                     int init_rows, int init_cols,
                                     int row_cord, int col_cord, int current_block_rows, int current_block_cols) {
	BoundaryBox top_mini_rect;
	Vector2 lowerbound, upperbound;  // cord of original data.
	float ratio = ori_level_image.cols / 16.0;
	int rect_size, rect_height, rect_width = 0;

	// For each grid-related 0-1 top_level_mask matrix, mapping their cord to original size.
	for (vector<Mat_<uchar>>::iterator it = top_level_mask.begin(); it != top_level_mask.end(); it++) {
		
		top_mini_rect = get_mini_rect(*it);

		if (top_mini_rect.GetHeight() == 0 || top_mini_rect.GetHeight() == 16 ||
				top_mini_rect.GetWidth() == 0 || top_mini_rect.GetWidth() == 16) {
			continue;
		}

		// Mapping to origin cord.
		lowerbound = top_mini_rect.LowerBound * ratio;
		upperbound = top_mini_rect.UpperBound * ratio;
		BoundaryBox ori_mini_rect(lowerbound, upperbound);

		// Get the right size area defined by users.
		// The default max_searchsize = 2048 (1/2 size of default block_size),
		// the default min_searchsize = 256 (depend on minimum 1 x 1 leaf node size of quadtree in resampled 16 x 16 area).
		rect_size = ori_mini_rect.GetSize();
		rect_height = ori_mini_rect.GetHeight();
		rect_width = ori_mini_rect.GetWidth();

		if ((rect_height <= max_searchsize) &&
			  (rect_width <= max_searchsize) &&
				(rect_size <= max_searchsize * max_searchsize)) {

      int current_row_cord = row_cord + ori_mini_rect.LBx();
      int current_col_cord = col_cord + ori_mini_rect.LBy();
      if (current_row_cord >= init_rows || current_col_cord >= init_cols) {
        continue;
      }

      if ((current_row_cord + rect_height) > init_rows) {
        rect_height = init_rows - current_row_cord;
      }

      if ((current_col_cord + rect_width) > init_cols) {
        rect_width = init_cols - current_col_cord;
      }

      search_patch_cord.first.push_back(current_row_cord);
      search_patch_cord.second.push_back(current_col_cord);
      search_patch_block_size.first.push_back(rect_height);
      search_patch_block_size.second.push_back(rect_width);
        
			//Mat image_object = ori_level_image(Range(ori_mini_rect.LBx(), ori_mini_rect.UBx()), Range(ori_mini_rect.LBy(), ori_mini_rect.UBy()));
			//Mat label_object = ori_level_label(Range(ori_mini_rect.LBx(), ori_mini_rect.UBx()), Range(ori_mini_rect.LBy(), ori_mini_rect.UBy()));

   //   // Ignore poor-information matrix.
   //   Mat gray_image_object;
   //   cvtColor(image_object, gray_image_object, COLOR_BGR2GRAY);
   //   double info_ratio = (double)countNonZero(gray_image_object) / (double)(gray_image_object.rows * gray_image_object.cols);
   //   if (info_ratio < 0.5) {
   //     continue;
   //   }

			//// For numpy, convert datatype to the CV_8U.
			//Mat image_object_8UC3(rect_height, rect_width, CV_8UC3);
			//image_object.copyTo(image_object_8UC3);

			//Mat label_object_8UC1(rect_height, rect_width, CV_8UC1);
			//label_object.copyTo(label_object_8UC1);

			//image_objects.push_back(image_object_8UC3);
			//label_objects.push_back(label_object_8UC1);
		}
	}
}

void QuadTree::delete_quadtree(QuadNode* root_node) {
	if (root_node == NULL) {
		return;
	}
	delete_quadtree(root_node->children[0]);
	delete_quadtree(root_node->children[1]);
	delete_quadtree(root_node->children[2]);
	delete_quadtree(root_node->children[3]);
	delete root_node;
	root_node = NULL;
	return;
}

//// Funtion used in 'quad_search'.
//void QuadTree::store_node(QuadNode* root_node) {
//	if (root_node == NULL) {
//		cout << "Empty root node.";
//		return;
//	}
//	if (!(root_node->has_children)) {
//		nodelist.push_back(root_node);
//	}
//	store_node(root_node->children[0]);
//	store_node(root_node->children[1]);
//	store_node(root_node->children[2]);
//	store_node(root_node->children[3]);
//}
//

//// Funtion used in 'quad_search'.
//void QuadTree::get_nodepath(QuadNode* root_node, QuadNode* random_node,
//														vector<QuadNode*>& nodepath, vector<QuadNode*>& temp_nodepath, int flag=0) {
//	if (root_node == NULL || flag) {
//		return;
//	}
//
//	temp_nodepath.push_back(root_node);
//	if (root_node == random_node)
//	{
//		flag = 1;
//		nodepath = temp_nodepath;
//	}
//	get_nodepath(root_node->children[0], random_node, nodepath, temp_nodepath, flag);
//	get_nodepath(root_node->children[1], random_node, nodepath, temp_nodepath, flag);
//	get_nodepath(root_node->children[2], random_node, nodepath, temp_nodepath, flag);
//	get_nodepath(root_node->children[3], random_node, nodepath, temp_nodepath, flag);
//
//	temp_nodepath.pop_back();
//}
//
//void QuadTree::quad_search() {
//	store_node(root_node);
//	// Random select the init quadnode area.
//	vector<QuadNode*> shuffle_nodelist(nodelist);
//	random_device rd;
//	mt19937 g(rd());
//	shuffle(begin(shuffle_nodelist), end(shuffle_nodelist), g);
//	//random_shuffle(shuffle_nodelist.begin(), shuffle_nodelist.end());
//
//	int n_nodes = kNodes;
//	if (shuffle_nodelist.size() < (unsigned int)n_nodes) {
//		n_nodes = shuffle_nodelist.size();
//	}
//
//	QuadNode* node_search_result;
//	vector<QuadNode*> nodepath, temp_nodepath;
//	for (int inode = 0; inode < n_nodes; inode++) {
//		int init_searchlabel = get_node_mainlabel(shuffle_nodelist[inode]);
//
//		// Store the path of root -> current node in nodepath.
//		get_nodepath(root_node, shuffle_nodelist[inode], nodepath, temp_nodepath, 0);
//
//		if (nodepath.empty()) {
//			cout << "Failed to search the path of root -> current node.";
//			return;
//		}
//		if (nodepath.size() == 1) {
//			allnode_quadsearch_result.push_back(nodepath[0]);
//			continue;
//		}
//
//		// Store max_size node area for init search node.
//		node_search_result = shuffle_nodelist[inode];
//		for (vector<QuadNode*>::reverse_iterator ipath = nodepath.rbegin() + 1; ipath != nodepath.rend(); ipath++) {
//			int main_label = get_node_mainlabel(*ipath);
//			if (main_label == init_searchlabel) {
//				node_search_result = *ipath;
//				if (node_search_result == nodepath[0]) {
//					allnode_quadsearch_result.push_back(node_search_result);
//					break;
//				}
//			}
//			else {
//				allnode_quadsearch_result.push_back(node_search_result);
//				break;
//			}
//		}
//	}
//}

//void QuadTree::grid_search(Mat& top_level_label, int n_classes, int ignore_label) {
//	for (vector<QuadNode*>::iterator it = allnode_quadsearch_result.begin(); it != allnode_quadsearch_result.end(); it++) {
//		/* Employ 0-1 matrix to store grid search information */
//		Mat_<uchar> istaken(512, 512, uchar(0));
//
//		int init_searchlabel = get_node_mainlabel(*it);
//		int size = (*it)->node_cord.GetHeight();
//
//		/* Ignore the orignal size node area. */
//		if (size == 512) {
//			continue;
//		}
//
//		/* 8-neighborhood grid search. */
//		const Vector2 dxy[8] = {
//				Vector2(-size, 0),
//				Vector2(-size, -size),
//				Vector2(0, -size),
//				Vector2(size, -size),
//				Vector2(size, 0),
//				Vector2(size, size),
//				Vector2(0, size),
//				Vector2(-size, size) };
//
//		vector<BoundaryBox> temp_grid_result;
//		BoundaryBox grid = (*it)->node_cord;
//		temp_grid_result.push_back(grid);
//		while (pop(temp_grid_result, grid)) {
//			istaken(Range(grid.LBx(), grid.UBx()), Range(grid.LBy(), grid.UBy())) = true;
//			for (int i = 0; i < 8; i++) {
//				Vector2 lowerbound = grid.LowerBound + dxy[i];
//				Vector2 upperbound = grid.UpperBound + dxy[i];
//				BoundaryBox grid(lowerbound, upperbound);
//
//				/* Search in 512x512 label area. */
//				if ((grid.LBx() >= 0 && grid.LBy() >= 0 && grid.LBx() < 512.0 && grid.LBy() < 512.0) &&
//					(grid.UBx() >= 0 && grid.UBy() >= 0 && grid.UBx() < 512.0 && grid.UBy() < 512.0)) {
//					if (istaken(Range(grid.LBx(), grid.UBx()), Range(grid.LBy(), grid.UBy())).at<uchar>(0, 0) == false) {
//						Mat grid_label = top_level_label(Range(grid.LBx(), grid.UBx()), Range(grid.LBy(), grid.UBy()));
//						int grid_mainlabel = get_grid_mainlabel(grid_label, n_classes, ignore_label);
//						if (grid_mainlabel == init_searchlabel) {
//							push(temp_grid_result, grid);
//						}
//					}
//				}
//			}
//		}
//		top_level_mask.push_back(istaken);
//	}
//}

//void QuadTree::get_multiscale_object(Mat& ori_level_image, Mat& ori_level_label, int max_searchsize, int min_searchsize) {
//	BoundaryBox top_mini_rect;
//	Vector2 lowerbound, upperbound;  // cord of original data.
//	float ratio = ori_level_image.cols / 16.0;
//	int rect_size, rect_height, rect_width = 0;
//
//	// For each grid-related 0-1 top_level_mask matrix, mapping their cord to original size.
//	for (vector<Mat_<uchar>>::iterator it = top_level_mask.begin(); it != top_level_mask.end(); it++) {
//		top_mini_rect = get_mini_rect(*it);
//		if (top_mini_rect.GetHeight() == 16 || top_mini_rect.GetHeight() == 0) {
//			continue;
//		}
//
//		lowerbound = top_mini_rect.LowerBound * ratio;
//		upperbound = top_mini_rect.UpperBound * ratio;
//		BoundaryBox ori_mini_rect(lowerbound, upperbound);
//
//		/* Get the right size area defined by users. */
//		rect_size = ori_mini_rect.GetSize();
//		rect_height = ori_mini_rect.GetHeight();
//		rect_width = ori_mini_rect.GetWidth();
//		if ((rect_size >= min_searchsize * min_searchsize && rect_size <= max_searchsize * max_searchsize) &&
//				(rect_height >= min_searchsize && rect_height <= max_searchsize) &&
//				(rect_width >= min_searchsize && rect_width <= max_searchsize)) {
//			Mat image_object = ori_level_image(Range(ori_mini_rect.LBx(), ori_mini_rect.UBx()), Range(ori_mini_rect.LBy(), ori_mini_rect.UBy()));
//			Mat label_object = ori_level_label(Range(ori_mini_rect.LBx(), ori_mini_rect.UBx()), Range(ori_mini_rect.LBy(), ori_mini_rect.UBy()));
//
//			/* For numpy, convert datatype to the CV_8U */
//			Mat image_object_8UC3(rect_height, rect_width, CV_8UC3);
//			image_object.copyTo(image_object_8UC3);
//			Mat label_object_8UC1(rect_height, rect_width, CV_8UC1);
//			label_object.copyTo(label_object_8UC1);
//
//			image_objects.push_back(image_object_8UC3);
//			label_objects.push_back(label_object_8UC1);
//		}
//	}
//}

//void QuadTree::get_multiscale_object(Mat& ori_level_image, Mat& ori_level_label, int max_searchsize, int min_searchsize) {
//	BoundaryBox top_mini_rect;
//	Vector2 lowerbound, upperbound;  // cord of original data.
//	float ratio = ori_level_image.cols / 16.0;
//	int rect_size, rect_height, rect_width = 0;
//
//	// For each grid-related 0-1 top_level_mask matrix, mapping their cord to original size.
//	for (vector<Mat_<uchar>>::iterator it = top_level_mask.begin(); it != top_level_mask.end(); it++) {
//		top_mini_rect = get_mini_rect(*it);
//		if (top_mini_rect.GetHeight() == 16 || top_mini_rect.GetHeight() == 0) {
//			continue;
//		}
//
//		lowerbound = top_mini_rect.LowerBound * ratio;
//		upperbound = top_mini_rect.UpperBound * ratio;
//		BoundaryBox ori_mini_rect(lowerbound, upperbound);
//
//		/* Get the right size area defined by users. */
//		rect_size = ori_mini_rect.GetSize();
//		rect_height = ori_mini_rect.GetHeight();
//		rect_width = ori_mini_rect.GetWidth();
//		if ((rect_size >= min_searchsize * min_searchsize && rect_size <= max_searchsize * max_searchsize) &&
//				(rect_height >= min_searchsize && rect_height <= max_searchsize) &&
//				(rect_width >= min_searchsize && rect_width <= max_searchsize)) {
//			Mat image_object = ori_level_image(Range(ori_mini_rect.LBx(), ori_mini_rect.UBx()), Range(ori_mini_rect.LBy(), ori_mini_rect.UBy()));
//			Mat label_object = ori_level_label(Range(ori_mini_rect.LBx(), ori_mini_rect.UBx()), Range(ori_mini_rect.LBy(), ori_mini_rect.UBy()));
//
//			/* For numpy, convert datatype to the CV_8U */
//			Mat image_object_8UC3(rect_height, rect_width, CV_8UC3);
//			image_object.copyTo(image_object_8UC3);
//			Mat label_object_8UC1(rect_height, rect_width, CV_8UC1);
//			label_object.copyTo(label_object_8UC1);
//
//			image_objects.push_back(image_object_8UC3);
//			label_objects.push_back(label_object_8UC1);
//		}
//	}
//}

}  // namespace luojianet_ms