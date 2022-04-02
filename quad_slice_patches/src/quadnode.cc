#include "quadnode.h"


QuadNode::QuadNode() {
	children[0] = NULL;
	children[1] = NULL;
	children[2] = NULL;
	children[3] = NULL;	
	this->node_cord = BoundaryBox(Vector2(0, 0), Vector2(0, 0));
	has_children = false;
}


QuadNode::QuadNode(BoundaryBox &node_cord) {
	children[0] = NULL;
	children[1] = NULL;
	children[2] = NULL;
	children[3] = NULL;
	this->node_cord = node_cord;
	has_children = false;
}


QuadNode::~QuadNode() {

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