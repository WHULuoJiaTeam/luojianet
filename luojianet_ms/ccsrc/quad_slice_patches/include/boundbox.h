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

#ifndef BOUNDARYBOX_H_
#define BOUNDARYBOX_H_

#include "Vector2.h"

namespace luojianet_ms {
// Implement basic quadtree boundarybox cordinate structure,
// inspired by https://github.com/bbbbyang/QuadTree-Segmentation.

//	struct of BoundaryBox.
//	  UL(0)	  |		 UR(3)
//	-----------------------
//	  LL(1)	  |		 LR(2)
class BoundaryBox {
 public:
	BoundaryBox();
	~BoundaryBox();

	/// \Initialize the boundaryBox area.
	/// \param[in] lower, LowerBound top and left coordinate of a boundary.
	/// \param[in] upper, UpperBound bottom and right coordinate of a boundary.
	BoundaryBox(Vector2 const, Vector2 const);

	// Return lowerboundary x coordinate.
	double LBx();

	// Return lowerboundary y coordinate.
	double LBy();

	// Return upperboundary x coordinate.
	double UBx();

	// Return upperboundary y coordinate.
	double UBy();

	// Return the UP quadrant of this rect area.
	BoundaryBox GetUR();

	// Return the UL quadrant of this rect area.
	BoundaryBox GetUL();

	// Return the LL quadrant of this rect area.
	BoundaryBox GetLL();

	// Return the LR quadrant of this rect area.
	BoundaryBox GetLR();

	// Return the square of the Euclidian distance between this Vector2 and another Vector2.
	double GetDims() const;

	// Return the width of this rectangle.
	int GetWidth() const;

	// Return the height of this rectangle.
	int GetHeight() const;

	// Return the size of this rectangle.
	int GetSize() const;

 public:
	Vector2 LowerBound; // bottom and left cord.
	Vector2 UpperBound; // top and right cord.
};

}  // namespace luojianet_ms

#endif	// BOUNDARYBOX_H_