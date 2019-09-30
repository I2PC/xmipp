/***************************************************************************
 *
 * Authors:     David Strelak (davidstrelak@gmail.com)
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

#ifndef XMIPP_LIBRARIES_DATA_POINT3D_H_
#define XMIPP_LIBRARIES_DATA_POINT3D_H_

#include "point.h"
#include "cuda_compatibility.h"
#include <initializer_list>

/** Class represents a point in 3D */
template <typename T>
class Point3D: Point {
public:
     CUDA_HD
     Point3D(T x = 0, T y = 0, T z = 0) :
             x(x), y(y), z(z) {
     }

    Point3D(const std::initializer_list<T> &l) {
        if (3 == l.size())
        {
            auto it = l.begin();
            x = *it++;
            y = *it++;
            z = *it++;
        }
    }

    T x;
    T y;
    T z;

    CUDA_H
    Point3D& operator/=(const T &rhs) const {
        return Point3D(x / rhs, y / rhs, z / rhs);
    }

    CUDA_H
    friend Point3D operator/(const Point3D &lhs, T rhs) {
        return lhs /= rhs;
    }
};

#endif /* XMIPP_LIBRARIES_DATA_POINT3D_H_ */
