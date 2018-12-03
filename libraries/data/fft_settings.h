/*
 * FFTSettings.h
 *
 *  Created on: Nov 19, 2018
 *      Author: dstrelak
 */

#ifndef FFTSETTINGS_H_
#define FFTSETTINGS_H_

template<typename T>
struct FFTSettings {
    FFTSettings(size_t x_spacial, size_t y, size_t z, size_t n, size_t batch,
            bool isInPlace) :
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

    size_t elemsSpacial() {
        return x_spacial * y * z * n;
    }

    size_t bytesSpacial() {
        return sizeof(T) * elemsSpacial();
    }

    size_t elemsFreq() {
        return x_freq * y * z * n;
    }

    size_t bytesFreq() {
        return sizeof(std::complex<T>) * elemsFreq();
    }
};

#endif /* FFTSETTINGS_H_ */
