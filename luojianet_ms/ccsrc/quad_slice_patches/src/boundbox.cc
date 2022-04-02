#include "boundbox.h"

// Default initialize the boundaryBox area
BoundaryBox::BoundaryBox() : LowerBound(0.0, 0.0), UpperBound(0.0, 0.0) {
}

/**
 * Initialize the boundaryBox area
 * @param LowerBound top and left coordinate of a boundary
 * @param UpperBound bottom and right coordinate of a boundary
 */
BoundaryBox::BoundaryBox(Vector2 const lower, Vector2 const upper) : LowerBound(lower), UpperBound(upper) {
}

// return lowerboundary x coordinate
double BoundaryBox::LBx() {
	return LowerBound.x;
}

// return lowerboundary y coordinate
double BoundaryBox::LBy() {
	return LowerBound.y;
}

// return upperboundary x coordinate
double BoundaryBox::UBx() {
	return UpperBound.x;
}

// return upperboundary y coordinate
double BoundaryBox::UBy() {
	return UpperBound.y;
}

// Get UR area boundaryBox parameters
BoundaryBox BoundaryBox::GetUR() {
	Vector2 lb;
	Vector2 ub;
	lb.set(LowerBound.x, (LowerBound.y + UpperBound.y) / 2.0);
	ub.set((LowerBound.x + UpperBound.x) / 2.0, UpperBound.y);
	return BoundaryBox(lb, ub);
}

// Get UL area boundaryBox parameters
BoundaryBox BoundaryBox::GetUL() {
	Vector2 lb;
	Vector2 ub;
	lb.set(LowerBound.x, LowerBound.y);
	ub.set((LowerBound.x + UpperBound.x) / 2.0, (LowerBound.y + UpperBound.y) / 2.0);
	return BoundaryBox(lb, ub);
}

// Get LL area boundaryBox parameters
BoundaryBox BoundaryBox::GetLL() {
	Vector2 lb;
	Vector2 ub;
	lb.set((LowerBound.x + UpperBound.x) / 2.0, LowerBound.y);
	ub.set(UpperBound.x, (LowerBound.y + UpperBound.y) / 2.0);
	return BoundaryBox(lb, ub);
}

// Get LR area boundaryBox parameters
BoundaryBox BoundaryBox::GetLR() {
	Vector2 lb;
	Vector2 ub;
	lb.set((LowerBound.x + UpperBound.x) / 2.0, (LowerBound.y + UpperBound.y) / 2.0);
	ub.set(UpperBound.x, UpperBound.y);
	return BoundaryBox(lb, ub);
}

// The condition to decide to subdivide or not
double BoundaryBox::GetDims() const {
	double dis;
	dis = UpperBound.getDistanceSquared(LowerBound);
	return dis;
}

// Get the width of the rectangle
int BoundaryBox::GetWidth() const {
	return (UpperBound.x - LowerBound.x);
}

// Get the height of the rectangle
int BoundaryBox::GetHeight() const {
	return (UpperBound.y - LowerBound.y);
}

// Get the size of the rectangle
int BoundaryBox::GetSize() const {
	int size;
	size = (UpperBound.x - LowerBound.x) * (UpperBound.y - LowerBound.y);
	return size;
}