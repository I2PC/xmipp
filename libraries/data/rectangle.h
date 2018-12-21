/*
 * rectangle.h
 *
 *  Created on: Dec 13, 2018
 *      Author: david
 */

#ifndef LIBRARIES_DATA_RECTANGLE_H_
#define LIBRARIES_DATA_RECTANGLE_H_

#include "dimensions.h"
#include "point2D.h"
#include <assert.h>
#include "template_helper.h"

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

    const T tl; // top left corner
    const T br; // bottom right corner
};



#endif /* LIBRARIES_DATA_RECTANGLE_H_ */
