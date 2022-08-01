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

#include "boundbox.h"

namespace luojianet_ms {

BoundaryBox::BoundaryBox() : LowerBound(0.0, 0.0), UpperBound(0.0, 0.0) {}

BoundaryBox::~BoundaryBox() {}

BoundaryBox::BoundaryBox(Vector2 const lower, Vector2 const upper) : LowerBound(lower), UpperBound(upper) {}

double BoundaryBox::LBx() {
	return LowerBound.x;
}

double BoundaryBox::LBy() {
	return LowerBound.y;
}

double BoundaryBox::UBx() {
	return UpperBound.x;
}

double BoundaryBox::UBy() {
	return UpperBound.y;
}

BoundaryBox BoundaryBox::GetUR() {
	Vector2 lb;
	Vector2 ub;
	lb.set(LowerBound.x, (LowerBound.y + UpperBound.y) / 2.0);
	ub.set((LowerBound.x + UpperBound.x) / 2.0, UpperBound.y);
	return BoundaryBox(lb, ub);
}

BoundaryBox BoundaryBox::GetUL() {
	Vector2 lb;
	Vector2 ub;
	lb.set(LowerBound.x, LowerBound.y);
	ub.set((LowerBound.x + UpperBound.x) / 2.0, (LowerBound.y + UpperBound.y) / 2.0);
	return BoundaryBox(lb, ub);
}

BoundaryBox BoundaryBox::GetLL() {
	Vector2 lb;
	Vector2 ub;
	lb.set((LowerBound.x + UpperBound.x) / 2.0, LowerBound.y);
	ub.set(UpperBound.x, (LowerBound.y + UpperBound.y) / 2.0);
	return BoundaryBox(lb, ub);
}

BoundaryBox BoundaryBox::GetLR() {
	Vector2 lb;
	Vector2 ub;
	lb.set((LowerBound.x + UpperBound.x) / 2.0, (LowerBound.y + UpperBound.y) / 2.0);
	ub.set(UpperBound.x, UpperBound.y);
	return BoundaryBox(lb, ub);
}

double BoundaryBox::GetDims() const {
	double dis;
	dis = UpperBound.getDistanceSquared(LowerBound);
	return dis;
}

int BoundaryBox::GetHeight() const {
	return (UpperBound.x - LowerBound.x);
}

int BoundaryBox::GetWidth() const {
	return (UpperBound.y - LowerBound.y);
}

int BoundaryBox::GetSize() const {
	int size;
	size = (UpperBound.x - LowerBound.x) * (UpperBound.y - LowerBound.y);
	return size;
}

}  // namespace luojianet_ms