/*
 * FFTSettings.h
 *
 *  Created on: Nov 19, 2018
 *      Author: dstrelak
 */

#ifndef FFTSETTINGS_H_
#define FFTSETTINGS_H_


const struct FFTSettings {
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
};


#endif /* FFTSETTINGS_H_ */
