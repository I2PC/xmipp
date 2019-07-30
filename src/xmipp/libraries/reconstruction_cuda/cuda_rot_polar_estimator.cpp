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

#include <cuda_runtime_api.h>
#include "reconstruction_cuda/cuda_asserts.h"
#include "cuda_rot_polar_estimator.h"
#include "cuda_gpu_polar.cu"

namespace Alignment {

template<typename T>
void CudaRotPolarEstimator<T>::init2D(const HW &hw) {
    try {
        m_gpu = &dynamic_cast<const GPU&>(hw);
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of GPU expected");
    }

    m_firstRing = this->m_dims->x() / 5;
    m_lastRing = (this->m_dims->x() - 1) / 2;
//    m_lastRing = (this->m_dims->x() - 3) / 2; // FIXME DS uncomment // so that we have some edge around the biggest ring
    // all rings have the same number of samples, to make FT easier
    m_samples = std::max(1, 2 * (int)(M_PI * m_lastRing)); // keep this even
    m_polarSettings = new FFTSettingsNew<T>(m_samples, getNoOfRings(), 1, this->m_dims->n(), this->m_batch);

    gpuErrchk(cudaMalloc(&m_d_batch_tmp1, std::max(
            this->m_dims->xy() * this->m_batch * sizeof(T), // Cartesian batch
            m_polarSettings->fBytesBatch()))); // FT of the samples
    gpuErrchk(cudaMalloc(&m_d_batch_tmp2, std::max(
            m_polarSettings->sBytesBatch(), // IFT of the samples
            m_polarSettings->fBytesBatch()))); // FT of the samples

    if (std::is_same<T, float>()) { // FIXME DS remove
        m_dataAux.resize(this->m_dims->y(), this->m_dims->x());
    }

    this->m_isInit = true;
}

template<typename T>
void CudaRotPolarEstimator<T>::release() {
    delete m_polarSettings;

    // device memory
    gpuErrchk(cudaFree(m_d_batch_tmp1));
    gpuErrchk(cudaFree(m_d_batch_tmp2));

    // FT plans
    CudaFFT<T>::release(m_singleToFD);
    CudaFFT<T>::release(m_batchToFD);
    CudaFFT<T>::release(m_batchToSD);


    m_dataAux.clear(); // FIXME DS remove
    m_refPolarFourierI.clear(); // FIXME DS remove
    delete m_refPlans; // FIXME DS remove
    m_rotCorrAux.clear(); // FIXME DS remove

    ARotationEstimator<T>::release();
    CudaRotPolarEstimator<T>::setDefault();
}

template<typename T>
void CudaRotPolarEstimator<T>::setDefault() {
    m_gpu = nullptr;
    m_polarSettings = nullptr;

    // device memory
    m_d_batch_tmp1 = nullptr;
    m_d_batch_tmp2 = nullptr;

    // FT plans
    m_singleToFD = nullptr;
    m_batchToFD = nullptr;
    m_batchToSD = nullptr;

    m_firstRing = -1;
    m_lastRing = -1;


    m_dataAux.clear(); // FIXME DS remove
    m_refPolarFourierI.clear(); // FIXME DS remove
    m_refPlans = nullptr; // FIXME DS remove
    m_rotCorrAux.clear(); // FIXME DS remove


    ARotationEstimator<T>::setDefault();
}

template<>
MultidimArray<double> CudaRotPolarEstimator<float>::convert(float *data) { // FIXME remove
    const size_t s = this->m_dims->xyz();
    for (size_t i = 0; i < s; ++i) {
        m_dataAux.data[i] = data[i];
    }
    return m_dataAux;
}

template<>
MultidimArray<double> CudaRotPolarEstimator<double>::convert(double *data) { // FIXME remove
    return MultidimArray<double>(
            this->m_dims->n(), this->m_dims->z(),
            this->m_dims->y(), this->m_dims->x(),
            data);
}

template<typename T> // FIXME DS rework
void CudaRotPolarEstimator<T>::load2DReferenceOneToN(const T *ref) {
    MultidimArray<double> tmp = convert(const_cast<T*>(ref));
    tmp.setXmippOrigin();
    normalizedPolarFourierTransform(tmp, m_refPolarFourierI, false,
            m_firstRing, m_lastRing, m_refPlans, 1);
    m_rotCorrAux.resize(2 * m_refPolarFourierI.getSampleNoOuterRing() - 1);
    m_aux.local_transformer.setReal(m_rotCorrAux);
    this->m_is_ref_loaded = true;
}


template<typename T>
void __attribute__((optimize("O0"))) CudaRotPolarEstimator<T>::computeRotation2DOneToN(T *h_others) {
    bool isReady = (this->m_isInit && (AlignType::OneToN == this->m_type));
    //&& this->m_is_ref_loaded); FIXME DS add this condition

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() and load reference");
    }

    this->m_rotations2D.reserve(this->m_dims->n());
    auto stream = *(cudaStream_t*)m_gpu->stream();
    // process signals in batches
    for (size_t offset = 0; offset < this->m_dims->n(); offset += this->m_batch) {
        // how many signals to process
        size_t toProcess = std::min(this->m_batch, this->m_dims->n() - offset);

        // copy memory
        gpuErrchk(cudaMemcpyAsync(
                m_d_batch_tmp1,
                h_others + offset * this->m_dims->xy(),
                toProcess * this->m_dims->xy() * sizeof(T),
                cudaMemcpyHostToDevice, stream));


        // call kernel
        dim3 dimBlock(32, 32);
        dim3 dimGrid(
            ceil(toProcess * m_samples / (float)dimBlock.x),
            ceil(getNoOfRings() / (float)dimBlock.y));
        polarFromCartesian<T, true>
            <<<dimGrid, dimBlock, 0, stream>>> (
            m_d_batch_tmp1, this->m_dims->x(), this->m_dims->y(),
            m_d_batch_tmp2, m_samples, getNoOfRings(), m_firstRing, toProcess);

        // FIXME DS rework eventually
        auto polars = new T[m_samples * getNoOfRings() * this->m_batch]();
        // copy data back
        gpuErrchk(cudaMemcpyAsync(
                polars,
                m_d_batch_tmp2,
                toProcess * m_samples * getNoOfRings() * sizeof(T),
                cudaMemcpyDeviceToHost, stream));
        MultidimArray<double> ma;
        for (size_t i = 0; i < toProcess; ++i) {
            Polar<double>tmp;
            for (size_t r = m_firstRing; r <= m_lastRing; ++r) { // rings are Y (rows)
                tmp.ring_radius.emplace_back(r);
                size_t row = r - m_firstRing;
                size_t offsetTmp = (i * getNoOfRings() * m_samples) + (row * m_samples);
                ma.resizeNoCopy(m_samples);
                for (size_t x = 0; x < m_samples; ++x) {
//                    printf("copying to %lu from %lu (val %f)\n", x, offsetTmp + x, polars[offsetTmp + x]);
                    ma.data[x] = (double) polars[offsetTmp + x];
                }
                tmp.rings.push_back(ma);
            }
            Polar<std::complex<double>> m_polarFourierI;
            Polar_fftw_plans *m_plans = nullptr;
            normalizedPolarFourierTransform(tmp, m_polarFourierI, true, m_plans);
            delete m_plans;
            this->m_rotations2D.emplace_back(
                best_rotation(m_refPolarFourierI, m_polarFourierI, m_aux));
        }
        delete[] polars;
    }
    this->m_is_rotation_computed = true;
}

template<typename T>
void CudaRotPolarEstimator<T>::check() {
    ARotationEstimator<T>::check();
    if (this->m_dims->x() != this->m_dims->y()) {
        // because of the rings
        REPORT_ERROR(ERR_ARG_INCORRECT, "This estimator can work only with square signal");
    }
//    if (this->m_dims->x() < 6) {
//        // we need some edge around the biggest ring, to avoid invalid memory access
//        REPORT_ERROR(ERR_ARG_INCORRECT, "The input signal is too small.");
//    } // FIXME DS uncomment
    if (this->m_dims->isPadded()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Padded signal is not supported");
    }
    if (m_polarSettings->sElemsBatch() > std::numeric_limits<int>::max()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Too big batch. It would cause int overflow in the cuda kernel");
    }
}

// explicit instantiation
template class CudaRotPolarEstimator<float>;
template class CudaRotPolarEstimator<double>;

} /* namespace Alignment */
