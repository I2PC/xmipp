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

#ifndef FFTSETTINGS_NEW_H_
#define FFTSETTINGS_NEW_H_

//#include <iosfwd>
#include "dimensions.h"
#include <cassert>
#include <complex>

template<typename T>
class FFTSettingsNew {
public:
    explicit FFTSettingsNew(size_t x, size_t y = 1, size_t z = 1, size_t n = 1,
            size_t batch = 1, // meaning no batch processing
            bool isInPlace = false) :
            m_spatial(Dimensions(x, y, z, n)),
            m_freq(x / 2 + 1, y, z, n),
            m_batch(batch),
            m_isInPlace(isInPlace) {
        assert(batch <= n);
        assert(batch > 0);
    };

    explicit FFTSettingsNew(const Dimensions &spatial,
            size_t batch = 1, // meaning no batch processing
            bool isInPlace = false) :
            m_spatial(spatial),
            m_freq(spatial.x() / 2 + 1, spatial.y(), spatial.z(), spatial.n()),
            m_batch(batch),
            m_isInPlace(isInPlace) {
        assert(batch <= spatial.n());
        assert(batch > 0);
    };

    inline constexpr Dimensions sDim() const {
        return m_spatial;
    }

    inline constexpr Dimensions fDim() const {
        return m_freq;
    }

    inline constexpr size_t batch() const {
        return m_batch;
    }

    inline constexpr size_t fBytesSingle() const {
        return m_freq.xyz() * sizeof(std::complex<T>);
    }

    inline constexpr size_t fBytes() const {
        return m_freq.size() * sizeof(std::complex<T>);
    }

    inline constexpr size_t fBytesBatch() const {
        return m_freq.xyz() * m_batch * sizeof(std::complex<T>);
    }

    inline constexpr size_t sBytesSingle() const {
        return m_spatial.xyz() * sizeof(T);
    }

    inline constexpr size_t sBytes() const {
        return m_spatial.size() * sizeof(T);
    }

    inline constexpr size_t sBytesBatch() const {
        return m_spatial.xyz() * m_batch * sizeof(T);
    }

private:
    Dimensions m_spatial;
    Dimensions m_freq;
    size_t m_batch;
    size_t m_isInPlace;
};

#endif /* FFTSETTINGS_NEW_H_ */
