/*
 * point2D.h
 *
 *  Created on: Dec 13, 2018
 *      Author: david
 */

#ifndef LIBRARIES_DATA_POINT2D_H_
#define LIBRARIES_DATA_POINT2D_H_

#include "point.h"

/** Struct represents a point in 2D */
template<typename T>
class Point2D: Point {
public:
    explicit Point2D(T x, T y) :
            x(x), y(y) {
    }
    ;
    const T x;
    const T y;

    Point2D& operator/=(T rhs) const {
        return Point2D(x / rhs, y / rhs);
    }

    friend Point2D operator/(const Point2D &lhs, T rhs) {
        return lhs /= rhs;
    }

    Point2D operator-=(T rhs) const {
        return Point2D(x - rhs, y - rhs);
    }

    Point2D operator+=(T rhs) const {
        return Point2D(x + rhs, y + rhs);
    }

    Point2D operator-=(const Point2D &rhs) const {
        return Point2D(x - rhs.x, y - rhs.y);
    }

    friend Point2D operator-(const Point2D &lhs, T rhs) {
        return lhs -= rhs;
    }

    friend Point2D operator+(const Point2D &lhs, T rhs) {
        return lhs += rhs;
    }

    friend Point2D operator-(const Point2D &lhs, const Point2D &rhs) {
        return lhs -= rhs;
    }
};



#endif /* LIBRARIES_DATA_POINT2D_H_ */
