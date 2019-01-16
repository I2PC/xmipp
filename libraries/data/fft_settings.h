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

#ifndef FFTSETTINGS_H_
#define FFTSETTINGS_H_

#include <iostream>
#include "dimensions.h"

template<typename T>
struct FFTSettings {
    explicit FFTSettings(size_t x, size_t y = 1, size_t z = 1, size_t n = 1,
            size_t batch = 0, bool isInPlace = false) :
            dim(x, y, z, n), x_freq(x / 2 + 1), batch(
                    batch), isInPlace(isInPlace) {
    }
    ;

    explicit FFTSettings(const Dimensions &d,
            size_t batch = 0, bool isInPlace = false) :
            dim(d), x_freq(d.x() / 2 + 1), batch(
                    batch), isInPlace(isInPlace) {
    }
    ;
    const size_t x_freq;
    const Dimensions dim;
    const size_t batch;
    const bool isInPlace;

    size_t elemsSpacial() const {
        return dim.size();
    }

    size_t bytesSpacial() const {
        return sizeof(T) * elemsSpacial();
    }

    size_t elemsFreq() const {
        return x_freq * dim.y() * dim.z() * dim.n() * 2; // * 2 for complex numbers
    }

    size_t bytesFreq() const {
        return sizeof(std::complex<T>) * elemsFreq();
    }

    friend std::ostream& operator<<(std::ostream &os,
            const FFTSettings<T> &s) {

        os << s.dim.x() << "(" << s.x_freq << ")" << " * " << s.dim.y() << " * "
                << s.dim.z() << " * " << s.dim.n() << ", batch: " << s.batch
                << ", inPlace: " << (s.isInPlace ? "yes" : "no");

        return os;
    }
};

#endif /* FFTSETTINGS_H_ */
