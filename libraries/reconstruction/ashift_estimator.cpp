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
#include "ashift_estimator.h"

namespace Alignment {

template<typename T>
std::vector<T> AShiftEstimator<T>::findMaxAroundCenter(
        const T *correlations,
        const Dimensions &dims,
        const Point2D<size_t> &maxShift,
        std::vector<Point2D<int>> &shifts) {
    size_t xHalf = dims.x() / 2;
    size_t yHalf = dims.y() / 2;

    assert(0 == shifts.size());
    assert(2 <= dims.x());
    assert(2 <= dims.y());
    assert(1 == dims.z());
    assert(nullptr != correlations);
    assert(maxShift.x <= xHalf);
    assert(maxShift.y <= yHalf);
    assert(0 < maxShift.x);
    assert(0 < maxShift.y);
    assert( ! dims.isPadded());

    auto result = std::vector<T>();
    shifts.reserve(dims.n());
    result.reserve(dims.n());

    size_t maxDist = maxShift.x * maxShift.y;
    for (size_t n = 0; n < dims.n(); ++n) {
        size_t offsetN = n * dims.xyz();
        // reset values
        size_t maxX;
        size_t maxY;
        T val = std::numeric_limits<T>::lowest();
        // iterate through the center
        for (size_t y = yHalf - maxShift.y; y <= yHalf + maxShift.y; ++y) {
            size_t offsetY = y * dims.x();
            int logicY = (int)y - yHalf;
            T ySq = logicY * logicY;
            for (size_t x = xHalf - maxShift.x; x <= xHalf + maxShift.x; ++x) {
                int logicX = (int)x - yHalf;
                // continue if the Euclidean distance is too far
                if ((ySq + (logicX * logicX)) > maxDist) continue;
                // get current value and update, if necessary
                T tmp = correlations[offsetN + offsetY + x];
                if (tmp > val) {
                    val = tmp;
                    maxX = logicX;
                    maxY = logicY;
                }
            }
        }
        // store results
        result.push_back(val);
        shifts.emplace_back(maxX, maxY);
    }
    return result;
}

template<typename T>
std::vector<T> AShiftEstimator<T>::findMaxAroundCenter(
        const T *correlations,
        const Dimensions &dims,
        size_t maxShift,
        std::vector<Point2D<int>> &shifts) {
    return findMaxAroundCenter(correlations, dims, Point2D<size_t>(maxShift, maxShift), shifts);
}

// explicit instantiation
template class AShiftEstimator<float>;
template class AShiftEstimator<double>;

} /* namespace Alignment */
