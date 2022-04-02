/*
* Created by: ZhangZhan
* Wuhan University
* zhangzhanstep@whu.edu.cn
* Copyright (c) 2021
*/

#ifndef QUADNODE_H
#define QUADNODE_H

#include "boundbox.h"

#include <vector>


class QuadNode {
public:
	bool has_children;
	BoundaryBox node_cord;
	QuadNode* children[4];
	std::vector<int> label_value;
public:
	QuadNode();
	~QuadNode();

	QuadNode(BoundaryBox &node_cord);
	void sub_quadtree();
};


#endif