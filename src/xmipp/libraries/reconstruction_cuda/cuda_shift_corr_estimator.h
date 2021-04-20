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
#include "cuda_fft.h"
#include "gpu.h"
#include <thread>
#include <condition_variable>

#include "cuda_single_extrema_finder.h"

namespace Alignment {

template<typename T>
class CudaShiftCorrEstimator : public AShiftCorrEstimator<T> {
public:
    CudaShiftCorrEstimator() {
        setDefault();
    }

    virtual ~CudaShiftCorrEstimator() {
        release();
    }

    void init2D(const std::vector<HW*> &hw, AlignType type, const FFTSettingsNew<T> &dims, size_t maxShift,
            bool includingBatchFT, bool includingSingleFT,
            bool allowDataOverwrite) override;

    void release();

    void load2DReferenceOneToN(const std::complex<T> *h_ref) override;

    void load2DReferenceOneToN(const T *h_ref) override;

    void computeCorrelations2DOneToN(
        std::complex<T> *h_inOut, bool center) override;

    void computeCorrelations2DOneToN(const HW &hw,
        std::complex<T> *inOut,
        const std::complex<T> *ref,
        const Dimensions &dims,
        bool center) override;

    void computeShift2DOneToN(T *others) override;

    static std::vector<Point2D<float>> computeShifts2DOneToN(
        const std::vector<GPU*> &gpus,
        std::complex<T> *d_othersF,
        T *d_othersS,
        std::complex<T> *d_ref,
        const FFTSettingsNew<T> &settings,
        cufftHandle plan,
        T *h_centers,
        size_t maxShift);

    template<bool center>
    static void sComputeCorrelations2DOneToN(
        const GPU &gpu,
        std::complex<T> *d_inOut,
        const std::complex<T> *d_ref,
        const Dimensions &dims);

    HW& getHW() const override {
        return *m_workStream;
    }

private:
    GPU *m_loadStream;
    GPU *m_workStream;

    // device memory
    std::complex<T> *m_d_single_FD;
    std::complex<T> *m_d_batch_FD;
    T *m_d_single_SD;
    T *m_d_batch_SD_work;
    T *m_d_batch_SD_load;

    // host memory
    T *m_h_centers;

    // synch primitives
    std::mutex *m_mutex;
    std::condition_variable *m_cv;
    bool m_isDataReady;

    // FT plans
    cufftHandle *m_singleToFD;
    cufftHandle *m_batchToFD;
    cufftHandle *m_batchToSD;

    // flags
    // bind the flag for host with the device flag
    bool &m_is_d_single_FD_loaded = AShiftCorrEstimator<T>::m_is_ref_FD_loaded;

    void init2DOneToN() override;
    void setDefault();
    void check() override;
    void loadThreadRoutine(T *others);
    void waitAndConvert();
    using AShiftEstimator<T>::init2D;
};


} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_CUDA_SHIFT_ALIGNER_H_ */

