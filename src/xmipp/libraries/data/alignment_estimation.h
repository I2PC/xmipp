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

#ifndef LIBRARIES_DATA_ALIGNMENT_ESTIMATION_H_
#define LIBRARIES_DATA_ALIGNMENT_ESTIMATION_H_

#include "core/matrix2d.h"

namespace Alignment {

struct AlignmentEstimation {
    AlignmentEstimation(size_t n) {
        poses.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            auto tmp = Matrix2D<double>();
            tmp.initIdentity(3);
            poses.emplace_back(tmp);
        }
        correlations.resize(n);
    }

    // This matrix describe the estimated transform, i.e. if you want to correct for the movement,
    // you have to inverse it
    std::vector<Matrix2D<double>> poses;
    std::vector<float> correlations;
};

} /* namespace Alignment */

#endif /* LIBRARIES_DATA_ALIGNMENT_ESTIMATION_H_ */
