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

class Dimensions {
public:
    explicit constexpr Dimensions(size_t x, size_t y = 1, size_t z = 1, size_t n = 1) :
            m_x(x), m_y(y), m_z(z), m_n(n) {
    }
    ;

    inline constexpr size_t x() const {
        return m_x;
    }

    inline constexpr size_t y() const {
        return m_y;
    }

    inline constexpr size_t z() const {
        return m_z;
    }

    inline constexpr size_t n() const {
        return m_n;
    }

    inline constexpr size_t xy() const {
        return m_x * m_y;
    }

    constexpr size_t size() const {
        return m_x * m_y * m_z * m_n;
    }

    friend std::ostream& operator<<(std::ostream &os, const Dimensions &d) {
        os << d.x() << " * " << d.y() << " * " << d.z() << " * " << d.n();
        return os;
    }

    constexpr bool operator==(const Dimensions &b) const {
        return (m_x == b.m_x)
                && (m_y == b.m_y)
                && (m_z == b.m_z)
                && (m_n == b.m_n);
    }

private:
    size_t m_x;
    size_t m_y;
    size_t m_z;
    size_t m_n;
};



#endif /* LIBRARIES_DATA_DIMENSIONS_H_ */
