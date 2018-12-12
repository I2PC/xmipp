/*
 * dimensions.h
 *
 *  Created on: Dec 12, 2018
 *      Author: david
 */

#ifndef LIBRARIES_DATA_DIMENSIONS_H_
#define LIBRARIES_DATA_DIMENSIONS_H_

#include <cstddef>

class Dimensions {
public:
    explicit Dimensions(size_t x, size_t y = 1, size_t z = 1, size_t n = 1) :
            x(x), y(y), z(z), n(n) {
    }
    ;

    constexpr size_t size() const {
        return x * y * z * n;
    }

    const size_t x;
    const size_t y;
    const size_t z;
    const size_t n;

    friend std::ostream& operator<<(std::ostream &os, const Dimensions &d) {
        os << d.x << " * " << d.y << " * " << d.z << " * " << d.n;
        return os;
    }
};



#endif /* LIBRARIES_DATA_DIMENSIONS_H_ */
