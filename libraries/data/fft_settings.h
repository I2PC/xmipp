/*
 * FFTSettings.h
 *
 *  Created on: Nov 19, 2018
 *      Author: dstrelak
 */

#ifndef FFTSETTINGS_H_
#define FFTSETTINGS_H_

#include <iostream>

template<typename T>
struct FFTSettings {
    FFTSettings(size_t x_spacial = 0, size_t y = 0, size_t z = 0, size_t n = 0,
            size_t batch = 0, bool isInPlace = false) :
            x_spacial(x_spacial), x_freq(x_spacial / 2 + 1), y(y), z(z), n(n), batch(
                    batch), isInPlace(isInPlace) {
    }
    ;
    size_t x_spacial;
    size_t x_freq;
    size_t y;
    size_t z;
    size_t n;
    size_t batch;
    bool isInPlace;

    size_t elemsSpacial() const {
        return x_spacial * y * z * n;
    }

    size_t bytesSpacial() const {
        return sizeof(T) * elemsSpacial();
    }

    size_t elemsFreq() const {
        return x_freq * y * z * n;
    }

    size_t bytesFreq() const {
        return sizeof(std::complex<T>) * elemsFreq();
    }

    friend std::ostream& operator<<(std::ostream &os,
            const FFTSettings<T> &s) {

        os << s.x_spacial << "(" << s.x_freq << ")" << " * " << s.y << " * "
                << s.z << " * " << s.n << ", batch: " << s.batch << ", inPlace:"
                << std::to_string(s.isInPlace) << std::endl;

        return os;
    }
};

#endif /* FFTSETTINGS_H_ */
