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

#include "shift_corr_estimator.h"

namespace Alignment {

template<typename T>
template<bool center>
void ShiftCorrEstimator<T>::computeCorrelations2DOneToN(
        std::complex<T> *inOut,
        const std::complex<T> *ref,
        size_t xDim, size_t yDim, size_t nDim) {
    if (center) {
        assert(0 == (xDim % 2));
        assert(0 == (yDim % 2));
    }
    for (size_t n = 0; n < nDim; ++n) {
        size_t offsetN = n * xDim * yDim;
        for (size_t y = 0; y < yDim; ++y) {
            size_t offsetY = y * xDim;
            for (size_t x = 0; x < xDim; ++x) {
                size_t destIndex = offsetN + offsetY + x;
                auto r = ref[offsetY + x];
                auto o = r * std::conj(inOut[destIndex]);
                inOut[destIndex] = o;
                if (center) {
                    int centerCoeff = 1 - 2 * ((x + y) & 1); // center FT, input must be even
                    inOut[destIndex] *= centerCoeff;
                }
            }
        }
    }
}

// explicit instantiation
template void ShiftCorrEstimator<float>::computeCorrelations2DOneToN<false>(std::complex<float>*,
        const std::complex<float>*,
        size_t xDim, size_t yDim, size_t nDim);
template class ShiftCorrEstimator<float>;

template void ShiftCorrEstimator<double>::computeCorrelations2DOneToN<false>(std::complex<double>*,
        const std::complex<double>*,
        size_t xDim, size_t yDim, size_t nDim);
template class ShiftCorrEstimator<double>;

} /* namespace Alignment */
