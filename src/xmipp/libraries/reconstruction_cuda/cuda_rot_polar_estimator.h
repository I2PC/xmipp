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

#ifndef LIBRARIES_RECONSTRUCTION_CUDA_CUDA_ROT_POLAR_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_CUDA_CUDA_ROT_POLAR_ESTIMATOR_H_

#include "reconstruction/arotation_estimator.cpp"
#include "reconstruction_cuda/gpu.h"
#include "cuda_fft.h"
#include "data/polar.h" // FIXME DS remove
#include <thread>
#include <condition_variable>

namespace Alignment {

template<typename T>
class CudaRotPolarEstimator : public ARotationEstimator<T> {
public :
    CudaRotPolarEstimator() {
        setDefault();
    }

    virtual ~CudaRotPolarEstimator() {
        release();
    }

    void release() override;

    void sComputeCorrelationsOneToN(
            const GPU &gpu,
            std::complex<T> *d_inOut,
            const std::complex<T> *d_ref,
            const Dimensions &dims,
            int firstRingRadius);

private:
    const GPU *m_gpu;
    int m_firstRing;
    int m_lastRing;
    int m_samples;

    FFTSettingsNew<T> *m_logicalSettings; // each signal is 2D array, rows are rings of samples
    FFTSettingsNew<T> *m_hwSettings; // we actually need to process signals * rows 1D samples
    FFTSettingsNew<T> *m_inverseSettings; // we actually need to process signals * N 1D samples

    // device memory
    std::complex<T> *m_d_ref;
    T *m_d_batch;
    T *m_d_batchPolarOrCorr;
    T *m_d_batchPolarFD;

    // FT plans
    cufftHandle *m_singleToFD;
    cufftHandle *m_batchToFD;
    cufftHandle *m_batchToSD;

    std::mutex *m_mutex;
    std::condition_variable *m_cv;
    bool m_isDataReady;

    void check() override;
    void setDefault() override;

    void init2D(const HW &hw) override;

    void load2DReferenceOneToN(const T *ref) override;

    void computeRotation2DOneToN(T *h_others) override;

    constexpr size_t getNoOfRings() const {
        return 1 + m_lastRing - m_firstRing;
    }

    MultidimArray<double> convert(T *data); // FIXME DS remove
    MultidimArray<double> m_dataAux; // FIXME DS remove
    Polar<std::complex<double>> m_refPolarFourierI; // FIXME DS remove
    Polar_fftw_plans *m_refPlans = nullptr; // FIXME DS remove
    MultidimArray<double> m_rotCorrAux; // FIXME DS remove
    RotationalCorrelationAux m_aux; // FIXME DS remove

    void loadThreadRoutine(T *h_others, void *loadStream);
};

}/* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_CUDA_CUDA_ROT_POLAR_ESTIMATOR_H_ */
