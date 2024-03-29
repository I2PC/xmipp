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

#ifndef LIBRARIES_DATA_DIMENSIONS_H_
#define LIBRARIES_DATA_DIMENSIONS_H_

#include <ostream>
#include "data/cuda_compatibility.h"

/**@defgroup Dimensions Dimensions
   @ingroup DataLibrary */
//@{
class Dimensions {
public:
    explicit constexpr Dimensions(size_t x, size_t y = 1, size_t z = 1, size_t n = 1,
            size_t pad_x = 0, size_t pad_y = 0, size_t pad_z = 0) :
            m_x(x), m_y(y), m_z(z), m_n(n),
            m_pad_x(pad_x), m_pad_y(pad_y), m_pad_z(pad_z) {}

    constexpr Dimensions createSingle() const {
        return Dimensions(m_x, m_y, m_z, 1, m_pad_x, m_pad_y, m_pad_z);
    }

    constexpr Dimensions copyForN(size_t n) const {
        return Dimensions(m_x, m_y, m_z, n, m_pad_x, m_pad_y, m_pad_z);
    }

    CUDA_HD
    constexpr size_t x() const {
        return m_x;
    }

    inline constexpr size_t xPadded() const {
        return m_x + m_pad_x;
    }

    CUDA_HD
    constexpr size_t y() const {
        return m_y;
    }

    inline constexpr size_t yPadded() const {
        return m_y + m_pad_y;
    }

    CUDA_HD
    constexpr size_t z() const {
        return m_z;
    }

    inline constexpr size_t zPadded() const {
        return m_z + m_pad_z;
    }

    CUDA_HD
    constexpr size_t n() const {
        return m_n;
    }

    inline constexpr size_t xy() const {
        return m_x * m_y;
    }

    inline constexpr size_t xyPadded() const {
        return (m_x + m_pad_x) * (m_y + m_pad_y);
    }

    CUDA_HD
    constexpr size_t xyz() const {
        return m_x * m_y * m_z;
    }

    inline constexpr size_t xyzPadded() const {
        return (m_x + m_pad_x) * (m_y + m_pad_y) * (m_z + m_pad_z);
    }

    CUDA_HD
    constexpr size_t sizeSingle() const {
        return xyz();
    }

    inline constexpr size_t size() const {
        return xyz() * m_n;
    }

    inline constexpr size_t sizePadded() const {
        return xyzPadded() * m_n;
    }

    friend std::ostream& operator<<(std::ostream &os, const Dimensions &d) {
        os << d.x() << " * " << d.y() << " * " << d.z() << " * " << d.n();
        return os;
    }

    inline constexpr bool equalSizesIgnorePadding(const Dimensions &b) const {
        return (m_x == b.m_x)
                && (m_y == b.m_y)
                && (m_z == b.m_z)
                && (m_n == b.m_n);
    }

    inline constexpr bool equalExceptNPadded(const Dimensions &b) const {
        return (m_x == b.m_x)
                && (m_y == b.m_y)
                && (m_z == b.m_z)
                // && (m_n == b.m_n) ignore on purpose
                && equalPadding(b);
    }

    inline constexpr bool equalPadding(const Dimensions &b) const {
        return (m_pad_x == b.m_pad_x)
                && (m_pad_y == b.m_pad_y)
                && (m_pad_z == b.m_pad_z);
    }

    inline constexpr bool isValid() const {
        return (0 != m_x)
                && (0 != m_y)
                && (0 != m_z)
                && (0 != m_n);
    }

    constexpr bool operator==(const Dimensions &b) const {
        return equalSizesIgnorePadding(b) && equalPadding(b);
    }

    constexpr bool operator!=(const Dimensions &b) const {
        return !(*this == b);
    }

    inline constexpr bool isPadded() const {
        return (0 != m_pad_x) || (0 != m_pad_y) || (0 != m_pad_z);
    }

    constexpr bool is1D() const {
        return (m_z == 1) && (m_y == 1);
    }

    CUDA_HD
    constexpr bool is2D() const {
        return (m_z == 1) && (m_y != 1);
    }

    CUDA_HD
    constexpr bool is3D() const {
        return (m_z != 1) && (m_y != 1);
    }

    CUDA_HD
    inline int getDimAsNumber() const {
        if (is3D()) return 3;
        if (is2D()) return 2;
        return 1;
    }

private:
    size_t m_x;
    size_t m_y;
    size_t m_z;
    size_t m_n;

    size_t m_pad_x;
    size_t m_pad_y;
    size_t m_pad_z;
};
//@}
#endif /* LIBRARIES_DATA_DIMENSIONS_H_ */
