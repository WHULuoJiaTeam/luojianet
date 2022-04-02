/*
* Created by: ZhangZhan
* Wuhan University
* zhangzhanstep@whu.edu.cn
* Copyright (c) 2021
*/

#ifndef BOUNDBOX_H
#define BOUNDBOX_H

#include "Vector2.h"

//	enumerate four quadrants
//	 UL(0)	    |		UR(3)
//	--------------------------
//	 LL(1)	    |		LR(2)


class BoundaryBox { // struct
public:
	Vector2 LowerBound; // bottom and left
	Vector2 UpperBound; // top and right
public:
	BoundaryBox();
	BoundaryBox(Vector2 const, Vector2 const);
	double LBx(); // return lowerboundary x coordinate
	double LBy(); // return lowerboundary y coordinate
	double UBx(); // return upperboundary x coordinate
	double UBy(); // return upperboundary y coordinate
	BoundaryBox GetUR(); // return the UP quadrant of this rect area
	BoundaryBox GetUL(); // return the UL quadrant of this rect area
	BoundaryBox GetLL(); // return the LL quadrant of this rect area
	BoundaryBox GetLR(); // return the LR quadrant of this rect area
	double GetDims() const; // return the square of the Euclidian distance between this Vector2 and another Vector2.
	int GetWidth() const; // return the width of this rectangle
	int GetHeight() const; // return the height of this rectangle
	int GetSize() const; // return the size of this rectangle
};


#endif