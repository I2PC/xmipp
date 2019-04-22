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

#ifndef LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_CUDA_SHIFT_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_CUDA_SHIFT_ESTIMATOR_H_

#include <type_traits>
#include "reconstruction/ashift_corr_estimator.h"
#include "data/fft_settings_new.h"
#include "core/xmipp_error.h"
#include "cuda_fft.h"
#include "gpu.h"

namespace Alignment {

template<typename T>
class CudaShiftCorrEstimator : public AShiftCorrEstimator<T> {
public:
    CudaShiftCorrEstimator() {
        setDefault();
    }

    ~CudaShiftCorrEstimator() {
        release();
    }

    void init2D(const GPU &gpu, AlignType type, const FFTSettingsNew<T> &dims, size_t maxShift=0,
            bool includingBatchFT=false, bool includingSingleFT=false);

    void release();

    void load2DReferenceOneToN(const std::complex<T> *h_ref);

    void load2DReferenceOneToN(const T *h_ref);

    template<bool center>
    void computeCorrelations2DOneToN(
        std::complex<T> *h_inOut);

    void computeCorrelations2DOneToN(const HW &hw,
        std::complex<T> *inOut,
        const std::complex<T> *ref,
        size_t xDim, size_t yDim, size_t nDim, bool center);

    std::vector<Point2D<int>> computeShift2DOneToN(
        T *h_others);

    static std::vector<Point2D<int>> computeShifts2DOneToN(
        const GPU &gpu,
        std::complex<T> *d_othersF,
        std::complex<T> *d_ref,
        size_t xDimF, size_t yDimF, size_t nDim,
        T *d_othersS, // this must be big enough to hold batch * centerSize^2 elements!
        cufftHandle plan,
        size_t xDimS,
        T *h_centers, size_t maxShift);

    template<bool center>
    static void sComputeCorrelations2DOneToN(
        const GPU &gpu,
        std::complex<T> *d_inOut,
        const std::complex<T> *d_ref,
        size_t xDim, size_t yDim, size_t nDim);

private:
    const FFTSettingsNew<T> *m_settingsInv;
    size_t m_maxShift;
    size_t m_centerSize;
    AlignType m_type;
    const GPU *m_gpu;

    // device memory
    std::complex<T> *m_d_single_FD;
    std::complex<T> *m_d_batch_FD;
    T *m_d_single_SD;
    T *m_d_batch_SD;

    // host memory
    T *m_h_centers;

    // FT plans
    cufftHandle *m_singleToFD;
    cufftHandle *m_batchToFD;
    cufftHandle *m_batchToSD;

    // flags
    bool m_includingBatchFT;
    bool m_includingSingleFT;
    bool m_isInit;
    bool m_is_d_single_FD_loaded;

    void check();
    void init2DOneToN();
    void setDefault();
};


} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_CUDA_SHIFT_ALIGNER_H_ */

