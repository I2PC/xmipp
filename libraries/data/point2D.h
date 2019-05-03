/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#ifndef LIBRARIES_DATA_POINT2D_H_
#define LIBRARIES_DATA_POINT2D_H_

#include "point.h"
#include <initializer_list>

/** Struct represents a point in 2D */
template<typename T>
class Point2D: Point {
public:
    explicit Point2D(T x, T y) :
            x(x), y(y) {
    }

    Point2D(const std::initializer_list<T> &l) {
        if (2 == l.size())
        {
            auto it = l.begin();
            x = *it++;
            y = *it++;
        }
    }

    T x; // FIXME DS this should be private member with setters / getters
    T y;


    Point2D operator/=(T rhs) const {
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

    Point2D operator+=(const Point2D &rhs) const {
        return Point2D(x + rhs.x, y + rhs.y);
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

    friend Point2D operator+(const Point2D &lhs, const Point2D &rhs) {
        return lhs += rhs;
    }

    friend Point2D operator-(const Point2D &lhs, const Point2D &rhs) {
        return lhs -= rhs;
    }
};

#endif /* LIBRARIES_DATA_POINT2D_H_ */
