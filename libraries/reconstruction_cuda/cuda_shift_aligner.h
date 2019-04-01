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
#include "reconstruction/ashift_aligner.h"
#include "data/fft_settings_new.h"
#include "core/xmipp_error.h"
#include "reconstruction_cuda/cuda_xmipp_utils.h"

namespace Alignment {

template<typename T>
class CudaShiftAligner : public AShiftAligner<T> {
public:
    CudaShiftAligner() : m_dims(0) {
        setDefault();
    }

    ~CudaShiftAligner() {
        release();
    }

    void init2D(AlignType type, const FFTSettingsNew<T> &dims, size_t maxShift=0, bool includingFT=false);

    void release();

    void load2DReferenceOneToN(const std::complex<T> *h_ref);

    void load2DReferenceOneToN(const T *h_ref);

    template<bool center>
    void computeCorrelations2DOneToN(
        std::complex<T> *h_inOut);

    std::vector<Point2D<T>> computeShift2DOneToN(
        T *h_others);

    static std::vector<Point2D<T>> computeShifts2DOneToN(
        std::complex<T> *d_othersF,
        std::complex<T> *d_ref,
        size_t xDimF, size_t yDimF, size_t nDim,
        T *d_othersS, // this must be big enough to hold batch * centerSize^2 elements!
        mycufftHandle handle,
        size_t xDimS,
        T *h_centers, MultidimArray<T> &helper, size_t maxShift);

    template<bool center>
    static void computeCorrelations2DOneToN(
        std::complex<T> *d_inOut,
        const std::complex<T> *d_ref,
        size_t xDim, size_t yDim, size_t nDim);

private:
    FFTSettingsNew<T> m_dims;
    size_t m_maxShift;
    size_t m_centerSize;
    AlignType m_type;

    // device memory
    std::complex<T> *m_d_single_FT; // FIXME rename to FD
    std::complex<T> *m_d_batch_FT;
    T *m_d_single_S; // FIXME rename to SD
    T *m_d_batch_S; // FIXME allocate this big enought to hold also the centers

    // host memory
    T *m_h_centers;
    MultidimArray<T> m_helper;
    T *m_origHelperData;

    // FT data
    mycufftHandle m_singleToFT;
    mycufftHandle m_batchToFT;
    mycufftHandle m_batchToSD;

    // flags
    bool m_includingFT;
    bool m_isInit;
    bool m_is_d_single_FT_loaded;

    void check();
    void init2DOneToN();
    void setDefault();
};


} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_CUDA_SHIFT_ALIGNER_H_ */
