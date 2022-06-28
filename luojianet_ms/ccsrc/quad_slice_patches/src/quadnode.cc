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

#include "quadnode.h"

namespace luojianet_ms {

QuadNode::QuadNode() {
	children[0] = NULL;
	children[1] = NULL;
	children[2] = NULL;
	children[3] = NULL;
	this->node_cord = BoundaryBox(Vector2(0, 0), Vector2(0, 0));
	has_children = false;
}

QuadNode::~QuadNode() {}

QuadNode::QuadNode(BoundaryBox& node_cord) {
	children[0] = NULL;
	children[1] = NULL;
	children[2] = NULL;
	children[3] = NULL;
	this->node_cord = node_cord;
	has_children = false;
}

void QuadNode::sub_quadtree() {
	BoundaryBox Box = node_cord;

	BoundaryBox URBox = Box.GetUR();
	BoundaryBox ULBox = Box.GetUL();
	BoundaryBox LLBox = Box.GetLL();
	BoundaryBox LRBox = Box.GetLR();

	has_children = true;

	children[0] = new QuadNode(ULBox);
	children[1] = new QuadNode(LLBox);
	children[2] = new QuadNode(LRBox);
	children[3] = new QuadNode(URBox);
}

}  // namespace luojianet_ms