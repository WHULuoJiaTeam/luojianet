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