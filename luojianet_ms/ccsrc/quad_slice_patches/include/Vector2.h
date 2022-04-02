//
//  Vector2.h
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

#ifndef VECTOR2_H
#define VECTOR2_H

#include <cassert>
#include <iostream>
#include <cmath>
#include <cfloat>



//
//  VECTOR2_IS_FINITE
//
//  A cross-platform macro that resolves into the appropriate
//    function to check if a floating point value is finite.
//    A floating point value is considered to be finite if it is
//    not any of the following:
//      -> positive infinity
//      -> negative infinity
//      -> NaN (not a number: 0/0)
//

#ifdef _WIN32
  // Microsoft non-standard function
#ifdef __MINGW32__
  #define _finite(v) (__builtin_isfinite(v))
#endif
#include <cfloat>
#define VECTOR2_IS_FINITE(n) _finite(n)
#elif __WIN32__
  // Microsoft non-standard function
#include <cfloat>
#define VECTOR2_IS_FINITE(n) _finite(n)
#else
  //  In theory, POSIX includes the isfinite macro defined in
  //    the C99 standard.  This macro is not included in any ISO
  //    C++ standard yet (as of January 2013).  However, this
  //    function does not seem to work.
  //#define VECTOR2_IS_FINITE(n) isfinite(n)
#define VECTOR2_IS_FINITE(n) true
#endif



//
//  Vector2
//
//  A class to store a math-style vector of length 2.  The 2
//    numbers that compose a Vector2 are refered to as its
//    elements and may be accesed using dot notation.  The
//    associated functions are all declared inline for speed
//    reasons.  In theory, this class should be as fast (or
//    faster, when using pass-by-reference) as using double
//    values, but more convenient.
//
//  The norm of a Vector2 is its "length", the distance along
//    it.
//

class Vector2
{
public:
	//
	//  Member Fields
	//
	//  These are the elements of the Vector2.  They can be queried
	//    and changed freely without disrupting the operation of the
	//    Vector2 instance.
	//

	double x;
	double y;

	//
	//  These are some standard Vector2s that may be useful
	//

	static const Vector2 ZERO;
	static const Vector2 ONE;
	static const Vector2 UNIT_X_PLUS;
	static const Vector2 UNIT_X_MINUS;
	static const Vector2 UNIT_Y_PLUS;
	static const Vector2 UNIT_Y_MINUS;

public:
	//
	//  Default Constructor
	//
	//  Purpose: To create a new Vector2 that is the zero vector.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: N/A
	//  Side Effect: A new Vector2 is created with elements
	//               (0.0, 0.0).
	//

	Vector2() : x(0.0), y(0.0)
	{}

	//
	//  Constructor
	//
	//  Purpose: To create a new Vector2 with the specified
	//           elements.
	//  Parameter(s):
	//    <1> x
	//    <2> y
	//    <3> z: The elements for the new Vector2
	//  Precondition(s): N/A
	//  Returns: N/A
	//  Side Effect: A new Vector2 is created with elements (x, y).
	//

	Vector2(double X, double Y) : x(X), y(Y)
	{}

	//
	//  Copy Constructor
	//
	//  Purpose: To create a new Vector2 with the same elements as
	//           an existing Vector2.
	//  Parameter(s):
	//    <1> original: The Vector2 to copy
	//  Precondition(s): N/A
	//  Returns: N/A
	//  Side Effect: A new Vector2 is created with the same elements
	//               as original.
	//

	Vector2(const Vector2& original) : x(original.x),
		y(original.y)
	{}

	//
	//  Destructor
	//
	//  Purpose: To safely destroy this Vector2 without memeory
	//           leaks.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: N/A
	//  Side Effect: All dynamically allocated memeory is freed.
	//

	~Vector2()
	{}

	//
	//  Assignment Operator
	//
	//  Purpose: To set the elements of this Vector2 to be equal to
	//           the elements of another.
	//  Parameter(s):
	//    <1> original: The Vector2 to copy
	//  Precondition(s): N/A
	//  Returns: A reference to this Vector2.
	//  Side Effect: The elements of this Vector2 are set to the
	//               elements of original.
	//

	Vector2& operator= (const Vector2& original)
	{
		//  Testing for self-assignment would take
		//    longer than just copying the values.
		x = original.x;
		y = original.y;

		return *this;
	}

	//
	//  Equality Operator
	//
	//  Purpose: To determine if this Vector2 is equal to another.
	//           Two Vector2s are equal IFF each of their elements
	//           are equal.
	//  Parameter(s):
	//    <1> other: The Vector2 to compare to
	//  Precondition(s): N/A
	//  Returns: Whether this Vector2 and other are equal.
	//  Side Effect: N/A
	//

	bool operator== (const Vector2& other) const
	{
		if (x != other.x) return false;
		if (y != other.y) return false;
		return true;
	}

	//
	//  Inequality Operator
	//
	//  Purpose: To determine if this Vector2 and another are
	//           unequal.  Two Vector2s are equal IFF each of their
	//           elements are equal.
	//  Parameter(s):
	//    <1> other: The Vector2 to compare to
	//  Precondition(s): N/A
	//  Returns: Whether this Vector2 and other are unequal.
	//  Side Effect: N/A
	//

	inline bool operator!= (const Vector2& other) const
	{
		if (x != other.x) return true;
		if (y != other.y) return true;
		return false;
	}

	//
	//  Negation Operator
	//
	//  Purpose: To create a new Vector2 that is the addative
	//           inverse of this Vector2.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: A Vector2 with elements (-x, -y).
	//  Side Effect: N/A
	//

	Vector2 operator- () const
	{
		return Vector2(-x, -y);
	}

	//
	//  Addition Operator
	//
	//  Purpose: To create a new Vector2 equal to the sum of this
	//           Vector2 and another.
	//  Parameter(s):
	//    <1> right: The other Vector2
	//  Precondition(s): N/A
	//  Returns: A Vector2 with elements (x + right.x, y + right.y).
	//  Side Effect: N/A
	//

	inline Vector2 operator+ (const Vector2& right) const
	{
		return Vector2(x + right.x, y + right.y);
	}

	//
	//  Subtraction Operator
	//
	//  Purpose: To create a new Vector2 equal to the differance of
	//           this Vector2 and another.
	//  Parameter(s):
	//    <1> right: The Vector2 to subtract from this Vector2
	//  Precondition(s): N/A
	//  Returns: A Vector2 with elements (x - other.x, y - other.y).
	//  Side Effect: N/A
	//

	Vector2 operator- (const Vector2& right) const
	{
		return Vector2(x - right.x, y - right.y);
	}

	//
	//  Multiplication Operator
	//
	//  Purpose: To create a new Vector2 equal to the product of
	//           this Vector2 and a scalar.
	//  Parameter(s):
	//    <1> factor: The scalar to multiply this Vector2 by
	//  Precondition(s): N/A
	//  Returns: A Vector2 with elements (x * factor, y * factor).
	//  Side Effect: N/A
	//

	Vector2 operator* (double factor) const
	{
		return Vector2(x * factor, y * factor);
	}

	//
	//  Division Operator
	//
	//  Purpose: To create a new Vector2 equal to this Vector2
	//           divided by a scalar.
	//  Parameter(s):
	//    <1> divisor: The scalar to divide this Vector2 by
	//  Precondition(s):
	//    <1> divisor != 0.0
	//  Returns: A Vector2 with elements (x / divisor, y / divisor).
	//  Side Effect: N/A
	//

	Vector2 operator/ (double divisor) const
	{
		assert(divisor != 0.0);

		return Vector2(x / divisor, y / divisor);
	}

	//
	//  Addition Assignment Operator
	//
	//  Purpose: To set this Vector2 to the sum of itself and
	//           another Vector2.
	//  Parameter(s):
	//    <1> right: The other Vector2
	//  Precondition(s): N/A
	//  Returns: N/A
	//  Side Effect: The elements of this Vector2 are set to
	//               (x + right.x, y + right.y).
	//

	Vector2& operator+= (const Vector2& right)
	{
		x += right.x;
		y += right.y;

		return *this;
	}

	//
	//  Subtraction Assignment Operator
	//
	//  Purpose: To set this Vector2 to the differance of itself and
	//           another Vector2.
	//  Parameter(s):
	//    <1> right: The Vector2 to subtract from this Vector2
	//  Precondition(s): N/A
	//  Returns: N/A
	//  Side Effect: The elements of this Vector2 are set to
	//               (x - right.x, y - right.y).
	//

	Vector2& operator-= (const Vector2& right)
	{
		x -= right.x;
		y -= right.y;

		return *this;
	}

	//
	//  Multiplication Assignment Operator
	//
	//  Purpose: To set this Vector2 to the product of itself and a
	//           scalar.
	//  Parameter(s):
	//    <1> factor: The scalar to multiply this Vector2 by
	//  Precondition(s): N/A
	//  Returns: N/A
	//  Side Effect: The elements of this Vector2 are set to
	//               (x * factor, y * factor).
	//

	Vector2& operator*= (double factor)
	{
		x *= factor;
		y *= factor;

		return *this;
	}

	//
	//  Division Assignment Operator
	//
	//  Purpose: To set this Vector2 to equal to the quotient of
	//           itself divided by a scalar.
	//  Parameter(s):
	//    <1> divisor: The scalar to divide this Vector2 by
	//  Precondition(s):
	//    <1> divisor != 0.0
	//  Returns: N/A
	//  Side Effect: The elements of this Vector2 are set to
	//               (x / divisor, y / divisor).
	//

	Vector2& operator/= (double divisor)
	{
		assert(divisor != 0.0);

		x /= divisor;
		y /= divisor;

		return *this;
	}

	//
	//  isFinite
	//
	//  Purpose: To determine if all components of this Vector2 are
	//           finite numbers.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: Whether this Vector2 has only finite components.
	//  Side Effect: N/A
	//

	bool isFinite() const
	{
		if (!VECTOR2_IS_FINITE(x)) return false;
		if (!VECTOR2_IS_FINITE(y)) return false;
		return true;
	}

	//
	//  isZero
	//
	//  Purpose: To determine if this Vector2 is the zero vector.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: Whether this Vector2 is equal to (0.0, 0.0).
	//  Side Effect: N/A
	//

	bool isZero() const
	{
		if (x != 0.0) return false;
		if (y != 0.0) return false;
		return true;
	}

	//
	//  isNormal
	//
	//  Purpose: To determine if this Vector2 is a unit vector.
	//           These functions require 12 significant digits (6
	//           for the square of the norm).  This function does
	//           the same thing as isUnit.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: Whether this Vector2 has a norm of 1.0.
	//  Side Effect: N/A
	//

	bool isNormal() const
	{
		double norm_sqr_minus_1 = getNormSquared() - 1;

		if (norm_sqr_minus_1 > 1e-6)
			return false;
		if (norm_sqr_minus_1 < -1e-6)
			return false;
		return true;
	}

	//
	//  isUnit
	//
	//  Purpose: To determine if this Vector2 is a unit vector.
	//           These functions require 12 significant digits (6
	//           for the square of the norm).  This function does
	//           the same thing as isNormal.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: Whether this Vector2 has a norm of 1.0.
	//  Side Effect: N/A
	//

	bool isUnit() const
	{
		double norm_sqr_minus_1 = getNormSquared() - 1;

		if (norm_sqr_minus_1 > 1e-6)
			return false;
		if (norm_sqr_minus_1 < -1e-6)
			return false;
		return true;
	}

	//
	//  getNorm
	//
	//  Purpose: To determine the norm of this Vector2.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: The norm of this Vector2.
	//  Side Effect: N/A
	//

	double getNorm() const
	{
		return sqrt(x * x + y * y);
	}

	//
	//  getNormSquared
	//
	//  Purpose: To determine the square of the norm of this
	//           Vector2.  This is significantly faster than
	//           calculating the norm itself.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: The square of the norm of this Vector2.
	//  Side Effect: N/A
	//

	double getNormSquared() const
	{
		return x * x + y * y;
	}

	//
	//  isNormLessThan
	//
	//  Purpose: To determine if the norm of this Vector2 is less
	//           than the specified value.  This is significantly
	//           faster than calculating the norm itself.
	//  Parameter(s):
	//    <1> length: The length to check against
	//  Precondition(s):
	//    <1> length >= 0.0
	//  Returns: Whether the norm of this Vector2 is less than
	//           length.
	//  Side Effect: N/A
	//

	double isNormLessThan(double length) const
	{
		assert(length >= 0.0);

		return getNormSquared() < length * length;
	}

	//
	//  isNormGreaterThan
	//
	//  Purpose: To determine if the norm of this Vector2 is greater
	//           than the specified value.  This is significantly
	//           faster than calculating the norm itself.
	//  Parameter(s):
	//    <1> length: The length to check against
	//  Precondition(s):
	//    <1> length >= 0.0
	//  Returns: Whether the norm of this Vector2 is greater than
	//           length.
	//  Side Effect: N/A
	//

	double isNormGreaterThan(double length) const
	{
		assert(length >= 0.0);

		return getNormSquared() > length * length;
	}

	//
	//  isNormLessThan
	//
	//  Purpose: To determine if the norm of this Vector2 is less
	//           than the norm of the specified Vector2.  This is
	//           significantly faster than calculating the norms
	//           themselves.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s): N/A
	//  Returns: Whether the norm of this Vector2 is less than the
	//           norm of other.
	//  Side Effect: N/A
	//

	double isNormLessThan(const Vector2& other) const
	{
		return getNormSquared() < other.getNormSquared();
	}

	//
	//  isNormGreaterThan
	//
	//  Purpose: To determine if the norm of this Vector2 is greater
	//           than the norm of the specified Vector2.  This is
	//           significantly faster than calculating the norms
	//           themselves.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s): N/A
	//  Returns: Whether the norm of this Vector2 is greater than
	//           the norm of other.
	//  Side Effect: N/A
	//

	double isNormGreaterThan(const Vector2& other) const
	{
		return getNormSquared() > other.getNormSquared();
	}

	//
	//  isAllComponentsNonZero
	//
	//  Purpose: To determine if all the elements of this Vector2
	//           are non-zero values.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: If any element of this Vector2 is equal to 0.0,
	//           false is returned.  Otherwise, true is returned.
	//  Side Effect: N/A
	//

	bool isAllComponentsNonZero() const
	{
		if (x == 0.0) return false;
		if (y == 0.0) return false;
		return true;
	}

	//
	//  isAllComponentsNonNegative
	//
	//  Purpose: To determine if all the elements of this Vector2
	//           are non-negative.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: If any element of this Vector2 is less than 0.0,
	//           false is returned.  Otherwise, true is returned.
	//  Side Effect: N/A
	//

	bool isAllComponentsNonNegative() const
	{
		if (x < 0.0) return false;
		if (y < 0.0) return false;
		return true;
	}

	//
	//  isAllComponentsPositive
	//
	//  Purpose: To determine if all the elements of this Vector2
	//           are positive.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: If any element of this Vector2 is less than or
	//           equal to 0.0, false is returned.  Otherwise, true
	//           is returned.
	//  Side Effect: N/A
	//

	bool isAllComponentsPositive() const
	{
		if (x <= 0.0) return false;
		if (y <= 0.0) return false;
		return true;
	}

	//
	//  isAllComponentsLessThan
	//
	//  Purpose: To determine if all components of this Vector2 are
	//           less than the corresponding components of
	//           another Vector2.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: Whether each component of this Vector2 is less than
	//           the corresponding component of other.
	//  Side Effect: N/A
	//

	bool isAllComponentsLessThan(const Vector2& other) const
	{
		assert(isFinite());
		assert(other.isFinite());

		if (x >= other.x) return false;
		if (y >= other.y) return false;
		return true;
	}

	//
	//  isAllComponentsLessThanOrEqual
	//
	//  Purpose: To determine if all components of this Vector2 are
	//           less than or equal to the corresponding
	//           components of another Vector2.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: Whether each component of this Vector2 is less than
	//           or equal to the corresponding component of
	//           other.
	//  Side Effect: N/A
	//

	bool isAllComponentsLessThanOrEqual(
		const Vector2& other) const
	{
		assert(isFinite());
		assert(other.isFinite());

		if (x > other.x) return false;
		if (y > other.y) return false;
		return true;
	}

	//
	//  isAllComponentsGreaterThan
	//
	//  Purpose: To determine if all components of this Vector2 are
	//           greater than the corresponding components of
	//           another Vector2.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: Whether each component of this Vector2 is greater
	//           than the corresponding component of other.
	//  Side Effect: N/A
	//

	bool isAllComponentsGreaterThan(
		const Vector2& other) const
	{
		assert(isFinite());
		assert(other.isFinite());

		if (x <= other.x) return false;
		if (y <= other.y) return false;
		return true;
	}

	//
	//  isAllComponentsGreaterThanOrEqual
	//
	//  Purpose: To determine if all components of this Vector2 are
	//           greater than or equal to the corresponding
	//           components of another Vector2.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: Whether each component of this Vector2 is greater
	//           than or equal to the corresponding component
	//           of other.
	//  Side Effect: N/A
	//

	bool isAllComponentsGreaterThanOrEqual(
		const Vector2& other) const
	{
		assert(isFinite());
		assert(other.isFinite());

		if (x < other.x) return false;
		if (y < other.y) return false;
		return true;
	}

	//
	//  isAllComponentsEqualTo
	//
	//  Purpose: To determine if all components of this Vector2 are
	//           equal to the specified value.
	//  Parameter(s):
	//    <1> value: The value to compare to
	//  Precondition(s):
	//    <1> isFinite()
	//  Returns: Whether each component of this Vector2 is equal to
	//           value.
	//  Side Effect: N/A
	//

	bool isAllComponentsEqualTo(double value) const
	{
		assert(isFinite());

		if (x != value) return false;
		if (y != value) return false;
		return true;
	}

	//
	//  isAllComponentsLessThan
	//
	//  Purpose: To determine if all components of this Vector2 are
	//           less than the specified value.
	//  Parameter(s):
	//    <1> value: The value to compare to
	//  Precondition(s):
	//    <1> isFinite()
	//  Returns: Whether each component of this Vector2 is less than
	//           value.
	//  Side Effect: N/A
	//

	bool isAllComponentsLessThan(double value) const
	{
		assert(isFinite());

		if (x >= value) return false;
		if (y >= value) return false;
		return true;
	}

	//
	//  isAllComponentsLessThanOrEqual
	//
	//  Purpose: To determine if all components of this Vector2 are
	//           less than or equal to the specified value.
	//  Parameter(s):
	//    <1> value: The value to compare to
	//  Precondition(s):
	//    <1> isFinite()
	//  Returns: Whether each component of this Vector2 is less than
	//           or equal to value.
	//  Side Effect: N/A
	//

	bool isAllComponentsLessThanOrEqual(double value) const
	{
		assert(isFinite());

		if (x > value) return false;
		if (y > value) return false;
		return true;
	}

	//
	//  isAllComponentsGreaterThan
	//
	//  Purpose: To determine if all components of this Vector2 are
	//           greater than the specified value.
	//  Parameter(s):
	//    <1> value: The value to compare to
	//  Precondition(s):
	//    <1> isFinite()
	//  Returns: Whether each component of this Vector2 is greater
	//           than value.
	//  Side Effect: N/A
	//

	bool isAllComponentsGreaterThan(double value) const
	{
		assert(isFinite());

		if (x <= value) return false;
		if (y <= value) return false;
		return true;
	}

	//
	//  isAllComponentsGreaterThanOrEqual
	//
	//  Purpose: To determine if all components of this Vector2 are
	//           greater than or equal to the specified value.
	//  Parameter(s):
	//    <1> value: The value to compare to
	//  Precondition(s):
	//    <1> isFinite()
	//  Returns: Whether each component of this Vector2 is greater
	//           than or equal to value.
	//  Side Effect: N/A
	//

	bool isAllComponentsGreaterThanOrEqual(
		double value) const
	{
		assert(isFinite());

		if (x < value) return false;
		if (y < value) return false;
		return true;
	}

	//
	//  getComponentX
	//
	//  Purpose: To create another Vector2 with the same X component
	//           as this Vector2 and the Y component set to 0.0.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: A copy of this Vector2 containing only the X
	//           component.
	//  Side Effect: N/A
	//

	Vector2 getComponentX() const
	{
		return Vector2(x, 0.0);
	}

	//
	//  getComponentY
	//
	//  Purpose: To create another Vector2 with the same Y component
	//           as this Vector2 and the X component set to 0.0.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: A copy of this Vector2 containing only the Y
	//           component.
	//  Side Effect: N/A
	//

	Vector2 getComponentY() const
	{
		return Vector2(0.0, y);
	}

	//
	//  getNormalized
	//
	//  Purpose: To create a normalized copy of this Vector2.
	//  Parameter(s): N/A
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> !isZero()
	//  Returns: A Vector2 with the same direction as this Vector2
	//           and a norm of 1.0.
	//  Side Effect: N/A
	//

	Vector2 getNormalized() const
	{
		assert(isFinite());
		assert(!isZero());

		assert(getNorm() != 0.0);
		double norm_ratio = 1.0 / getNorm();
		return Vector2(x * norm_ratio, y * norm_ratio);
	}

	//
	//  getNormalizedSafe
	//
	//  Purpose: To create a normalized copy of this Vector2 without
	//           crashing if this Vector2 is the zero vector.  This
	//           function is slower than the getNormalized function.
	//  Parameter(s): N/A
	//  Precondition(s):
	//    <1> isFinite()
	//  Returns: If this Vector2 is the zero vector, the zero vector
	//           is returned.  Otherwise, a Vector2 with the same
	//           direction as this Vector2 and a norm of 1.0 is
	//           returned.
	//  Side Effect: N/A
	//

	Vector2 getNormalizedSafe() const
	{
		assert(isFinite());

		if (isZero())
			return Vector2();

		assert(getNorm() != 0.0);
		double norm_ratio = 1.0 / getNorm();
		return Vector2(x * norm_ratio, y * norm_ratio);
	}

	//
	//  getCopyWithNorm
	//
	//  Purpose: To create a Vector2 with the same direction as this
	//           Vector2 and the specified norm.
	//  Parameter(s):
	//    <1> norm: The new norm
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> !isZero()
	//    <3> norm >= 0.0
	//  Returns: A Vector2 with the same direction as this Vector2
	//           and a norm of norm.
	//  Side Effect: N/A
	//

	Vector2 getCopyWithNorm(double norm) const
	{
		assert(isFinite());
		assert(!isZero());
		assert(norm >= 0.0);

		assert(getNorm() != 0.0);
		double norm_ratio = norm / getNorm();
		return Vector2(x * norm_ratio, y * norm_ratio);
	}

	//
	//  getCopyWithNormSafe
	//
	//  Purpose: To create a Vector2 with the same direction as this
	//           Vector2 and the specified norm.  This function will
	//           not crash if this Vector2 is the zero vector, but
	//           is slower than the getCopyWithNorm function.
	//  Parameter(s):
	//    <1> norm: The new norm
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> norm >= 0.0
	//  Returns: If this Vector2 is the zero vector, the zero vector
	//           is returned.  Otherwise, a Vector2 with the same
	//           direction as this Vector2 and a norm of norm is
	//           returned.
	//  Side Effect: N/A
	//

	Vector2 getCopyWithNormSafe(double norm) const
	{
		assert(isFinite());
		assert(norm >= 0.0);

		if (isZero())
			return Vector2();
		assert(getNorm() != 0.0);
		double norm_ratio = norm / getNorm();
		return Vector2(x * norm_ratio, y * norm_ratio);
	}

	//
	//  getTruncated
	//
	//  Purpose: To create a Vector2 with the same direction as this
	//           Vector2 and a norm no greater than the specified
	//           value.
	//  Parameter(s):
	//    <1> norm: The new maximum norm
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> norm >= 0.0
	//  Returns: If this Vector2 has a norm greater than norm, a
	//           Vector2 with the same direction as this Vector2 and
	//           a norm of norm is returned.  Otherwise, a copy of
	//           this Vector2 is returned.
	//  Side Effect: N/A
	//

	Vector2 getTruncated(double norm)
	{
		assert(isFinite());
		assert(norm >= 0.0);

		if (isNormGreaterThan(norm))
		{
			double norm_ratio = norm / getNorm();
			return Vector2(x * norm_ratio, y * norm_ratio);
		}
		else
			return Vector2(*this);
	}

	//
	//  set
	//
	//  Purpose: To change this Vector2 to have the specified
	//           elements.
	//  Parameter(s):
	//    <1> X
	//    <2> Y: The new elements for this Vector2
	//  Precondition(s): N/A
	//  Returns: N/A
	//  Side Effect: This Vector2 is set to (X, Y).
	//

	void set(double X, double Y)
	{
		x = X;
		y = Y;
	}

	//
	//  setZero
	//
	//  Purpose: To change this Vector2 to be the zero vector.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: N/A
	//  Side Effect: This Vector2 is set to (0.0, 0.0).
	//

	void setZero()
	{
		x = 0.0;
		y = 0.0;
	}

	//
	//  normalize
	//
	//  Purpose: To change this Vector2 have a norm of 1.0.
	//  Parameter(s): N/A
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> !isZero()
	//  Returns: N/A
	//  Side Effect: This Vector2 is set to have a norm of 1.0.  The
	//               direction of this Vector2 is unchanged.
	//

	void normalize()
	{
		assert(isFinite());
		assert(!isZero());

		assert(getNorm() != 0.0);
		double norm_ratio = 1.0 / getNorm();

		x *= norm_ratio;
		y *= norm_ratio;

		assert(isNormal());
	}

	//
	//  normalizeSafe
	//
	//  Purpose: To change this Vector2 have a norm of 1.0 if it is
	//           not the zero vector.  This function is slower than
	//           the normalize function.
	//  Parameter(s): N/A
	//  Precondition(s):
	//    <1> isFinite()
	//  Returns: N/A
	//  Side Effect: If this Vector2 is not the zero vector, it is
	//               set to have a norm of 1.0.  Otherwise, there is
	//               no effect.  In either case, the direction of
	//               this Vector2 is unchanged.
	//

	void normalizeSafe()
	{
		assert(isFinite());

		if (!isZero())
		{
			assert(getNorm() != 0.0);
			double norm_ratio = 1.0 / getNorm();

			x *= norm_ratio;
			y *= norm_ratio;
		}

		assert(isNormal());
	}

	//
	//  setNorm
	//
	//  Purpose: To change the norm of this Vector2.
	//  Parameter(s):
	//    <1> norm: The new norm
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> !isZero()
	//    <3> norm >= 0.0
	//  Returns: N/A
	//  Side Effect: This Vector2 is set to have a norm of norm.
	//               The direction of this Vector2 is unchanged.
	//

	void setNorm(double norm)
	{
		assert(isFinite());
		assert(!isZero());
		assert(norm >= 0.0);

		assert(getNorm() != 0.0);
		double norm_ratio = norm / getNorm();

		x *= norm_ratio;
		y *= norm_ratio;

		assert(getNormSquared() - (norm * norm) < 1e-6);
		assert(getNormSquared() - (norm * norm) > -1e-6);
	}

	//
	//  setNormSafe
	//
	//  Purpose: To change the norm of this Vector2 if it is not the
	//           zero vector.  This function is slower than the
	//           normalize function.
	//  Parameter(s):
	//    <1> norm: The new norm
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> norm >= 0.0
	//  Returns: N/A
	//  Side Effect: If this Vector2 is not the zero vector, it is
	//               set to have a norm of norm.  Otherwise, there
	//               is no effect.  In either case, the direction of
	//               this Vector2 is unchanged.
	//

	void setNormSafe(double norm)
	{
		assert(isFinite());
		assert(norm >= 0.0);

		if (!isZero())
		{
			assert(getNorm() != 0.0);
			double norm_ratio = norm / getNorm();

			x *= norm_ratio;
			y *= norm_ratio;
		}

		assert(getNormSquared() - (norm * norm) < 1e-6);
		assert(getNormSquared() - (norm * norm) > -1e-6);
	}

	//
	//  truncate
	//
	//  Purpose: To reduce the norm of this Vector2 to the specified
	//           if it is currently greater.
	//  Parameter(s):
	//    <1> norm: The new maximum norm
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> norm >= 0.0
	//  Returns: N/A
	//  Side Effect: If this Vector2 has a norm greater than norm,
	//               is set to have a norm of norm.  Otherwise there
	//               is no effect.  In either case, the direction of
	//               this Vector2 is unchanged.
	//

	void truncate(double norm)
	{
		assert(isFinite());
		assert(norm >= 0.0);

		if (isNormGreaterThan(norm))
		{
			double norm_ratio = norm / getNorm();

			x *= norm_ratio;
			y *= norm_ratio;

			assert(getNormSquared() - (norm * norm) < 1e-6);
			assert(getNormSquared() - (norm * norm) > -1e-6);
		}
	}

	//
	//  componentProduct
	//
	//  Purpose: To calculate the component-wise product of this
	//           Vector2 and another Vector2.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: A Vector2 with elements (x * other.x, y * other.y).
	//  Side Effect: N/A
	//

	Vector2 componentProduct(const Vector2& other)
	{
		assert(isFinite());
		assert(other.isFinite());

		return Vector2(x * other.x, y * other.y);
	}

	//
	//  componentRatio
	//
	//  Purpose: To calculate the component-wise ratio of this
	//           Vector2 and another Vector2.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//    <3> other.isAllComponentsNonZero()
	//  Returns: A Vector2 with elements (x / other.x, y / other.y).
	//  Side Effect: N/A
	//

	Vector2 componentRatio(const Vector2& other)
	{
		assert(isFinite());
		assert(other.isFinite());
		assert(other.isAllComponentsNonZero());

		return Vector2(x / other.x, y / other.y);
	}

	//
	//  componentRatioSafe
	//
	//  Purpose: To calculate the component-wise ratio of this
	//           Vector2 and another Vector2 without crashing
	//           if one of the elements of the second Vector2
	//           is zero.  This function is slower than
	//           componentRatio.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: A Vector2 with elements (x / other.x, y / other.y).
	//           If either of the elements in other is zero,
	//           the corresponding element of this Vector2 is
	//           returned for that element instead of a ratio.
	//  Side Effect: N/A
	//

	Vector2 componentRatioSafe(const Vector2& other)
	{
		assert(isFinite());
		assert(other.isFinite());

		return Vector2((other.x != 0.0) ? (x / other.x) : x,
			(other.y != 0.0) ? (y / other.y) : y);
	}

	//
	//  dotProduct
	//
	//  Purpose: To determine the dot/scaler/inner product of this
	//           Vector2 and another Vector2.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: *this (dot) other.
	//  Side Effect: N/A
	//

	double dotProduct(const Vector2& other) const
	{
		assert(isFinite());
		assert(other.isFinite());

		return x * other.x + y * other.y;
	}

	//
	//  getDistance
	//
	//  Purpose: To determine the Euclidian distance between this
	//           Vector2 and another Vector2.  If you only need to
	//           determine if the distance to another Vector2 is
	//           less or greater than some value, consider using one
	//           of the isDistanceLessThan or isDistanceGreaterThan
	//           functions.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: The Euclidean distance between this Vector2 and
	//           other.
	//  Side Effect: N/A
	//

	double getDistance(const Vector2& other) const
	{
		double diff_x = x - other.x;
		double diff_y = y - other.y;
		return sqrt(diff_x * diff_x + diff_y * diff_y);
	}

	//
	//  getDistanceSquared
	//
	//  Purpose: To determine the square of the Euclidian distance
	//           between this Vector2 and another Vector2.  This
	//           function is significantly faster than getDistance().
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: The square of the Euclidian distance between this
	//           Vector2 and other.
	//  Side Effect: N/A
	//

	double getDistanceSquared(const Vector2& other) const
	{
		double diff_x = x - other.x;
		double diff_y = y - other.y;
		return diff_x * diff_x + diff_y * diff_y;
	}

	//
	//  isDistanceLessThan
	//
	//  Purpose: To determine if the Euclidian distance between this
	//           Vector2 and another Vector2 is less than the
	//           specified value.  This function is significantly
	//           faster than getDistance().
	//  Parameter(s):
	//    <1> other: The other Vector2
	//    <2> distance: The cutoff distance
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: Whether the Euclidian distance between Vector2 and
	//           other is less than distance.
	//  Side Effect: N/A
	//

	bool isDistanceLessThan(const Vector2& other,
		double distance) const
	{
		return (getDistanceSquared(other) <
			distance * distance);
	}

	//
	//  isDistanceGreaterThan
	//
	//  Purpose: To determine if the Euclidian distance between this
	//           Vector2 and another Vector2 is greater than the
	//           specified value.  This function is significantly
	//           faster than getDistance().
	//  Parameter(s):
	//    <1> other: The other Vector2
	//    <2> distance: The cutoff distance
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: Whether the Euclidian distance between Vector2 and
	//           other is greater than distance.
	//  Side Effect: N/A
	//

	bool isDistanceGreaterThan(const Vector2& other,
		double distance) const
	{
		return (getDistanceSquared(other) >
			distance * distance);
	}

	//
	//  getManhattenDistance
	//
	//  Purpose: To determine the Manhatten distance between this
	//           Vector2 and another Vector2.  This is the sum of
	//           the differences between corresponding components.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: The Manhatten distance between this Vector2 and
	//           other.
	//  Side Effect: N/A
	//

	double getManhattenDistance(const Vector2& other) const
	{
		return fabs(x - other.x) + fabs(y - other.y);
	}

	//
	//  getChessboardDistance
	//
	//  Purpose: To determine the chessboard distance between this
	//           Vector2 and another Vector2.  This is the largest
	//           differences between corresponding components.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: The chessboard distance between this Vector2 and
	//           other.
	//  Side Effect: N/A
	//

	double getChessboardDistance(
		const Vector2& other) const
	{
		double dx = fabs(x - other.x);
		double dy = fabs(y - other.y);

		return (dx < dy) ? dy : dx;
	}

	//
	//  projection
	//
	//  Purpose: To determione the projection of this Vector2 onto
	//           another Vector2.
	//  Parameter(s):
	//    <1> project_onto: The Vector2 to be projected onto 
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//    <3> !project_onto.isZero()
	//  Returns: The projection of this Vector2 onto project_onto.
	//           This is the component of this Vector2 with the same
	//           (or opposite) direction as project_onto.
	//  Side Effect: N/A
	//

	Vector2 projection(const Vector2& project_onto) const
	{
		assert(isFinite());
		assert(project_onto.isFinite());
		assert(!project_onto.isZero());

		double norm = dotProduct(project_onto) /
			project_onto.getNormSquared();

		return project_onto * norm;
	}

	//
	//  getCosAngle
	//
	//  Purpose: To determine the cosine of the angle between this
	//           Vector2 and another Vector2.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> !isZero()
	//    <3> other.isFinite()
	//    <4> !other.isZero()
	//  Returns: The cosine of the angle between this Vector2 and
	//           other.
	//  Side Effect: N/A
	//

	double getCosAngle(const Vector2& other) const;

	//
	//  getCosAngleSafe
	//
	//  Purpose: To determine the cosine of the angle between this
	//           Vector2 and another Vector2, without crashing if
	//           one of the Vector2s is zero.  This function is
	//           slower than getCosAngle.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: The cosine of the angle between this Vector2 and
	//           other.  If either vector is zero, cos(0) = 1 is
	//           returned.
	//  Side Effect: N/A
	//

	double getCosAngleSafe(const Vector2& other) const;

	//
	//  getAngle
	//
	//  Purpose: To determine the angle in radians between this
	//           Vector2 and another Vector2.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> !isZero()
	//    <3> other.isFinite()
	//    <4> !other.isZero()
	//  Returns: The angle in radians between this Vector2 and
	//           other.
	//  Side Effect: N/A
	//

	double getAngle(const Vector2& other) const;

	//
	//  getAngleSafe
	//
	//  Purpose: To determine the angle in radians between this
	//           Vector2 and another Vector2, without crashing if
	//           one of the Vector2s is zero.  This function is
	//           slower than getAngle.
	//  Parameter(s):
	//    <1> other: The other Vector2
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> other.isFinite()
	//  Returns: The angle in radians between this Vector2 and
	//           other.  If either vector is zero, 0 is returned.
	//  Side Effect: N/A
	//

	double getAngleSafe(const Vector2& other) const;

	//
	//  getRotation
	//
	//  Purpose: To determine how far this Vector2 is rotated in
	//           radians, assuming the original Vector2 was facing
	//           in the X+ direction.
	//  Parameter(s): N/A
	//  Precondition(s):
	//    <1> isFinite()
	//    <2> !isZero()
	//  Returns: The current rotation of this Vector2 in radians.
	//  Side Effect: N/A
	//

	double getRotation() const
	{
		assert(isFinite());
		assert(!isZero());

		return atan2(y, x);
	}

	//
	//  getRotated
	//
	//  Purpose: To create a copy of this Vector2 by the specified
	//           angle in radians.
	//  Parameter(s):
	//    <1> radians: The angle to rotate in radians
	//  Precondition(s):
	//    <1> isFinite()
	//  Returns: A copy of this Vector2 rotated radians radians.
	//  Side Effect: N/A
	//

	Vector2 getRotated(double radians) const
	{
		assert(isFinite());

		double sin_angle = sin(radians);
		double cos_angle = cos(radians);

		return Vector2(cos_angle * x - sin_angle * y,
			sin_angle * x + cos_angle * y);
	}

	//
	//  rotate
	//
	//  Purpose: To rotate this Vector2 by the specified angle in
	//           radians.
	//  Parameter(s):
	//    <1> radians: The angle to rotate in radians
	//  Precondition(s):
	//    <1> isFinite()
	//  Returns: A copy of this Vector2 rotated radians radians.
	//  Side Effect: N/A
	//

	void rotate(double radians)
	{
		assert(isFinite());

		double sin_angle = sin(radians);
		double cos_angle = cos(radians);

		set(cos_angle * x - sin_angle * y,
			sin_angle * x + cos_angle * y);
	}

	//
	//  getRandomUnitVector
	//
	//  Purpose: To generate a Vector2 of norm 1 and with a uniform
	//           random direction.
	//  Parameter(s): N/A
	//  Precondition(s): N/A
	//  Returns: A uniform random unit vector.
	//  Side Effect: N/A
	//

	static Vector2 getRandomUnitVector();

	//
	//  getClosestPointOnLine
	//
	//  Purpose: To determine the point on a specified line segment
	//           closest to the specified point.
	//  Parameter(s):
	//    <1> l1
	//    <2> l2: The two ends of the line segment
	//    <3> p: The point
	//    <4> bounded: Whether the solution must line between the
	//                 ends of the line segment
	//  Precondition(s):
	//    <1> l1.isFinite()
	//    <2> l2.isFinite()
	//    <3> p.isFinite()
	//    <4> l1 != l2
	//  Returns: The point on the line from l1 to l2 that is closest
	//           to point p.  If bounded == true, the point returned
	//           will lie between or on points l1 and l2.
	//           Otherwise, the point returned may lie anywhere
	//           along the line l1 and l2 define.
	//  Side Effect: N/A
	//

	static Vector2 getClosestPointOnLine(const Vector2& l1,
		const Vector2& l2,
		const Vector2& p,
		bool bounded);
};

//
//  Stream Insertion Operator
//
//  Purpose: To print the specified Vector2 to the specified
//           output stream.
//  Parameter(s):
//    <1> r_os: The output stream
//    <2> vector: The Vector2
//  Precondition(s): N/A
//  Returns: A reference to r_os.
//  Side Effect: vector is printed to r_os.
//

std::ostream& operator<< (std::ostream& r_os,
	const Vector2& vector);

//
//  Multiplication Operator
//
//  Purpose: To create a new Vector2 equal to the product of
//           the specified scalar and the specified Vector2.
//  Parameter(s):
//    <1> scalar: The scalar
//    <2> vector: The Vector2
//  Precondition(s): N/A
//  Returns: A Vector2 with elements
//           (vector.x * scalar, vector.y * scalar).
//  Side Effect: N/A
//

inline Vector2 operator* (double scalar, const Vector2& vector)
{
	return Vector2(vector.x * scalar, vector.y * scalar);
}



#endif
