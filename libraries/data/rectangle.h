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

#ifndef LIBRARIES_DATA_RECTANGLE_H_
#define LIBRARIES_DATA_RECTANGLE_H_

#include "dimensions.h"
#include "point2D.h"
#include <assert.h>

template<typename T>
class Rectangle {
public:
    explicit Rectangle(T &topLeftCorner, T &bottomRightCorner) :
            tl(std::move(topLeftCorner)), br(std::move(bottomRightCorner)) {
        static_assert(std::is_base_of<Point, T>::value, "T must inherit from Point");
    }
    ;

    constexpr T getCenter() const {
        return (tl + br) / 2;
    }

    constexpr T getSize() const {
        return (br - tl) + 1;
    }

    const T tl; // top left corner
    const T br; // bottom right corner
};

#endif /* LIBRARIES_DATA_RECTANGLE_H_ */
