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

#include "dimensions.h"
#include <cassert>
#include <complex>

/**@defgroup FFTSettings FFTSettings
   @ingroup DataLibrary */
//@{

// Forward declare both templates (otherwise you'll get: warning: friend declaration  declares a non-template function)
template <typename T> class FFTSettings;
template <typename T> std::ostream& operator<<( std::ostream&, const FFTSettings<T>& );

template<typename T>
class FFTSettings {
public:
    explicit FFTSettings(size_t x, size_t y = 1, size_t z = 1, size_t n = 1,
            size_t batch = 1, // meaning no batch processing
            bool isInPlace = false,
            bool isForward = true) :
            m_spatial(x, y, z, n),
            m_freq(x / 2 + 1, y, z, n),
            m_batch(batch),
            m_isInPlace(isInPlace),
            m_isForward(isForward) {
        if (isInPlace) {
            size_t padding = (m_freq.x() * 2) - x;
            m_spatial = Dimensions(x, y, z, n, padding);
        }
        assert(batch <= n);
        assert(batch > 0);
    };

    explicit FFTSettings(const Dimensions &spatial,
            size_t batch = 1, // meaning no batch processing
            bool isInPlace = false,
            bool isForward = true) :
            m_spatial(spatial),
            m_freq(spatial.x() / 2 + 1, spatial.y(), spatial.z(), spatial.n()),
            m_batch(batch),
            m_isInPlace(isInPlace),
            m_isForward(isForward) {
        if (isInPlace) {
            size_t padding = (m_freq.x() * 2) - spatial.x();
            m_spatial = Dimensions(spatial.x(), spatial.y(), spatial.z(), spatial.n(), padding);
        }
        assert(batch <= spatial.n());
        assert(batch > 0);
    };

    constexpr Dimensions sDim() const {
        return m_spatial;
    }

    constexpr Dimensions fDim() const {
        return m_freq;
    }

    constexpr size_t batch() const {
        return m_batch;
    }

    constexpr size_t fBytesSingle() const {
        return m_freq.xyzPadded() * sizeof(std::complex<T>);
    }

    constexpr size_t fBytes() const {
        return m_freq.sizePadded() * sizeof(std::complex<T>);
    }

    constexpr size_t fBytesBatch() const {
        return m_freq.xyzPadded() * m_batch * sizeof(std::complex<T>);
    }

    constexpr size_t fElemsBatch() const {
        return m_freq.xyzPadded() * m_batch;
    }

    constexpr size_t sBytesSingle() const {
        return m_spatial.xyzPadded() * sizeof(T);
    }

    constexpr size_t sBytes() const {
        return m_spatial.sizePadded() * sizeof(T);
    }

    constexpr size_t sBytesBatch() const {
        return m_spatial.xyzPadded() * m_batch * sizeof(T);
    }

    constexpr size_t sElemsBatch() const {
        return m_spatial.xyzPadded() * m_batch;
    }

    constexpr bool isForward() const {
        return m_isForward;
    }

    constexpr bool isInPlace() const {
        return m_isInPlace;
    }

    constexpr size_t maxBytesBatch() const {
        return sBytesBatch() + (m_isInPlace ? 0 :fBytesBatch());
    }

    inline FFTSettings<T> createInverse() const {
        auto copy = FFTSettings<T>(*this);
        copy.m_isForward = ! this->m_isForward;
        return copy;
    }

    inline FFTSettings<T> createSingle() const {
        auto copy = FFTSettings<T>(m_spatial.x(), m_spatial.y(), m_spatial.z(), 1, 1,
                this->isInPlace(), this->isForward());
        return copy;
    }

    inline FFTSettings<T> createBatch() const {
        auto copy = FFTSettings<T>(m_spatial.x(), m_spatial.y(), m_spatial.z(), m_batch, m_batch,
                this->isInPlace(), this->isForward());
        return copy;
    }

    inline FFTSettings<T> createSubset(size_t n) const {
        assert(n <= m_spatial.n());
        auto copy = FFTSettings<T>(m_spatial.x(), m_spatial.y(), m_spatial.z(), n, n,
                this->isInPlace(), this->isForward());
        return copy;
    }

    inline FFTSettings<T> copyForBatch(size_t b) const {
        assert(b <= m_spatial.n());
        auto copy = FFTSettings<T>(m_spatial.x(), m_spatial.y(), m_spatial.z(), m_spatial.n(), b,
                this->isInPlace(), this->isForward());
        return copy;
    }

    friend std::ostream &operator<< <T>(std::ostream &os,
                                const FFTSettings<T> &s);

private:
    Dimensions m_spatial;
    Dimensions m_freq;
    size_t m_batch;
    bool m_isInPlace;
    bool m_isForward;
};


//@}
#endif /* FFTSETTINGS_H_ */
