/*
 * FFTSettings.h
 *
 *  Created on: Nov 19, 2018
 *      Author: dstrelak
 */

#ifndef FFTSETTINGS_H_
#define FFTSETTINGS_H_

#include <iostream>
#include "dimensions.h"

template<typename T>
struct FFTSettings {
    FFTSettings(size_t x, size_t y = 1, size_t z = 1, size_t n = 1,
            size_t batch = 0, bool isInPlace = false) :
            dim(x, y, z, n), x_freq(x / 2 + 1), batch(
                    batch), isInPlace(isInPlace) {
    }
    ;
    const size_t x_freq;
    const Dimensions dim;
    size_t batch;
    bool isInPlace;

    size_t elemsSpacial() const {
        return dim.size();
    }

    size_t bytesSpacial() const {
        return sizeof(T) * elemsSpacial();
    }

    size_t elemsFreq() const {
        return x_freq * dim.y * dim.z * dim.n;
    }

    size_t bytesFreq() const {
        return sizeof(std::complex<T>) * elemsFreq();
    }

    friend std::ostream& operator<<(std::ostream &os,
            const FFTSettings<T> &s) {

        os << s.dim.x << "(" << s.x_freq << ")" << " * " << s.dim.y << " * "
                << s.dim.z << " * " << s.dim.n << ", batch: " << s.batch
                << ", inPlace:"
                << std::to_string(s.isInPlace) << std::endl;

        return os;
    }
};

#endif /* FFTSETTINGS_H_ */
