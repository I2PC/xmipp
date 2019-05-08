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

#ifndef LIBRARIES_RECONSTRUCTION_ASHIFT_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_ASHIFT_ESTIMATOR_H_

#include "data/point2D.h"
#include "data/dimensions.h"
#include <vector>
#include <cassert>
#include <limits>

namespace Alignment {

enum class AlignType { None, OneToN, NToM, Consecutive };

template<typename T>
class AShiftEstimator {
public:
    static std::vector<T> findMaxAroundCenter(
            const T *data,
            const Dimensions &dims,
            const Point2D<size_t> &maxShift,
            std::vector<Point2D<int>> &shifts);

    static std::vector<T> findMaxAroundCenter(
            const T *data,
            const Dimensions &dims,
            size_t maxShift,
            std::vector<Point2D<int>> &shifts);
    virtual void release() = 0;
};

} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_ASHIFT_ESTIMATOR_H_ */
