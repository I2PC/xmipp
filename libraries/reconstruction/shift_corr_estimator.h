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

#ifndef LIBRARIES_RECONSTRUCTION_SHIFT_CORR_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_SHIFT_CORR_ESTIMATOR_H_

#include "ashift_corr_estimator.h"

namespace Alignment {

template<typename T>
class ShiftCorrEstimator : public AShiftCorrEstimator<T> {
public:
    template<bool center>
    static void computeCorrelations2DOneToN(
        std::complex<T> *inOut,
        const std::complex<T> *ref,
        size_t xDim, size_t yDim, size_t nDim);
};

}  /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_SHIFT_CORR_ESTIMATOR_H_ */
