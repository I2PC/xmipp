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

#ifndef LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_CUDA_SHIFT_ALIGNER_H_
#define LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_CUDA_SHIFT_ALIGNER_H_

#include <type_traits>
#include <vector>
//#include "reconstruction/ashift_aligner.h"
#include "data/fft_settings_new.h"
#include "core/xmipp_error.h"
#include "data/point2D.h"
#include "data/filters.h"
#include "reconstruction_cuda/cuda_xmipp_utils.h"
//#include "reconstruction_adapt_cuda/cuda_compatibility.h"

namespace Alignment {

template<typename T>
class CudaShiftAligner {
public:
    static std::vector<Point2D<T>> computeShift2DOneToN(
        T *h_others,
        T *h_ref,
        FFTSettingsNew<T> &dims,
        size_t maxShift);

    static std::vector<Point2D<T>> computeShift2DOneToN(
        std::complex<T> *h_others,
        std::complex<T> *h_ref,
        FFTSettingsNew<T> &dims,
        size_t maxShift);

    static std::vector<Point2D<T>> computeShifts2DOneToN(
        std::complex<T> *d_othersF,
        std::complex<T> *d_ref,
        size_t xDimF, size_t yDimF, size_t nDim,
        T *d_othersS, mycufftHandle handle,
        size_t xDimS,
        T *h_centers, MultidimArray<T> &helper, size_t maxShift);

    static std::vector<Point2D<T>> computeShiftFromCorrelations2D(
        T *h_centers, MultidimArray<T> &helper, size_t nDim,
        size_t centerSize, size_t maxShift);

    static void computeCorrelations2DOneToN(
        std::complex<T> *h_inOut,
        const std::complex<T> *h_ref,
        const FFTSettingsNew<T> &dims);

    template<bool center>
    static void computeCorrelations2DOneToN(
        std::complex<T> *d_inOut,
        const std::complex<T> *d_ref,
        size_t xDim, size_t yDim, size_t nDim);

};


} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_CUDA_SHIFT_ALIGNER_H_ */
