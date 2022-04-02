//
//  Vector2.cpp
//
//  This file is part of the ObjLibrary, by Richard Hamilton,
//    which is copyright Hamilton 2009-2014.
//
//  You may use these files for any purpose as long as you do
//    not explicitly claim them as your own work or object to
//    other people using them.  If you are in a position of
//    authority, you may forbid others ffrom using them in areas
//    that fall under your authority.  For example, a professor
//    could forbid students from using them for a class project,
//    or an employer could forbid employees using for a company
//    project.
//
//  If you are destributing the source files, you must not
//    remove this notice.  If you are only destributing compiled
//    code, no credit is required.
//
//  A (theoretically) up-to-date version of the ObjLibrary can
//    be found at:
//  http://infiniplix.ca/resources/obj_library/
//

#include <iostream>
#include <cstdlib>	// for rand()
#include <cassert>

#include "Vector2.h"

using namespace std;



const Vector2 Vector2::ZERO(0.0, 0.0);
const Vector2 Vector2::ONE(1.0, 1.0);
const Vector2 Vector2::UNIT_X_PLUS(1.0, 0.0);
const Vector2 Vector2::UNIT_X_MINUS(-1.0, 0.0);
const Vector2 Vector2::UNIT_Y_PLUS(0.0, 1.0);
const Vector2 Vector2::UNIT_Y_MINUS(0.0, -1.0);



double Vector2::getCosAngle(const Vector2& other) const
{
	assert(isFinite());
	assert(!isZero());
	assert(other.isFinite());
	assert(!other.isZero());

	double ratio = dotProduct(other) / (getNorm() * other.getNorm());

	//  In theory, ratio should always be in the range [-1, 1].
	//    Sadly, in reality there are floating point errors.
	return (ratio < -1.0) ? -1.0 : ((ratio > 1.0) ? 1.0 : ratio);
}

double Vector2::getCosAngleSafe(const Vector2& other) const
{
	assert(isFinite());
	assert(other.isFinite());

	if (isZero() || other.isZero())
		return 1.0;

	double ratio = dotProduct(other) / (getNorm() * other.getNorm());

	//  In theory, ratio should always be in the range [-1, 1].
	//    Sadly, in reality there are floating point errors.
	return (ratio < -1.0) ? -1.0 : ((ratio > 1.0) ? 1.0 : ratio);
}

double Vector2::getAngle(const Vector2& other) const
{
	assert(isFinite());
	assert(!isZero());
	assert(other.isFinite());
	assert(!other.isZero());

	static const double PI = 3.1415926535897932384626433832795;

	double ratio = dotProduct(other) / (getNorm() * other.getNorm());

	//  In theory, ratio should always be in the range [-1, 1].
	//    Sadly, in reality there are floating point errors.
	return (ratio < -1.0) ? PI : ((ratio > 1.0) ? 0.0 : acos(ratio));
}

double Vector2::getAngleSafe(const Vector2& other) const
{
	assert(isFinite());
	assert(other.isFinite());

	static const double PI = 3.1415926535897932384626433832795;

	if (isZero() || other.isZero())
		return 1.0;

	double ratio = dotProduct(other) / (getNorm() * other.getNorm());

	//  In theory, ratio should always be in the range [-1, 1].
	//    Sadly, in reality there are floating point errors.
	return (ratio < -1.0) ? PI : ((ratio > 1.0) ? 0.0 : acos(ratio));
}



Vector2 Vector2::getRandomUnitVector()
{
	static const double TWO_PI = 6.283185307179586476925286766559;

	double random0to1 = (double)rand() / ((double)(RAND_MAX)+1.0);
	double angle = random0to1 * TWO_PI;
	return Vector2(cos(angle), sin(angle));
}



Vector2 Vector2::getClosestPointOnLine(const Vector2& l1, const Vector2& l2, const Vector2& p, bool bounded)
{
	assert(l1.isFinite());
	assert(l2.isFinite());
	assert(p.isFinite());
	assert(l1 != l2);

	//
	//         + p
	//       /
	//     /
	//   +-----+-------+
	//  l1     s       l2
	//     +
	//     O
	//
	//  O: The origin (0, 0, 0)
	//  l1, l2: The two ends of the line segment
	//  p: The not-on-line-point
	//  s: The point on the line segment closest to p
	//

	Vector2 line_direction = l2 - l1;
	Vector2 p_direction = p - l1;
	Vector2 s_minus_l1 = p_direction.projection(line_direction);

	if (bounded)
	{
		if (s_minus_l1.dotProduct(line_direction) <= 0)
			return l1;
		else if (s_minus_l1.getNormSquared() > line_direction.getNormSquared())
			return l2;
	}

	return s_minus_l1 + l1;
}



ostream& operator<< (ostream& r_os, const Vector2& vector)
{
	r_os << "(" << vector.x << ", " << vector.y << ")";
	return r_os;
}