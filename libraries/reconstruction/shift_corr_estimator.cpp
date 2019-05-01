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
void ShiftCorrEstimator<T>::init2D(const HW &hw, AlignType type,
        const FFTSettingsNew<T> &settings, size_t maxShift,
        bool includingBatchFT, bool includingSingleFT) {
    release();
    try {
        m_cpu = &dynamic_cast<const CPU&>(hw);
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of CPU expected");
    }

    AShiftCorrEstimator<T>::init2D(type, settings, maxShift,
        includingBatchFT, includingSingleFT);

    this->m_isInit = true;
}

template<typename T>
void ShiftCorrEstimator<T>::load2DReferenceOneToN(const T *ref) {
    auto isReady = (this->m_isInit && (AlignType::OneToN == this->m_type) && this->m_includingSingleFT);
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to load a reference signal");
    }

    // perform FT
    FFTwT<T>::fft(m_singleToFD, ref, m_single_FD);

    // update state
    this->m_is_ref_FD_loaded = true;
}

template<typename T>
void ShiftCorrEstimator<T>::init2DOneToN() {
    AShiftCorrEstimator<T>::init2DOneToN();

    // allocate plans and space for data in Fourier domain
    m_batchToSD = FFTwT<T>::createPlan(*m_cpu, *this->m_settingsInv);
    auto settingsForw = this->m_settingsInv->createInverse();
    if (this->m_includingBatchFT) {
        m_batch_FD = new std::complex<T>[this->m_settingsInv->fElemsBatch()];
        m_batchToFD = FFTwT<T>::createPlan(*m_cpu, settingsForw);
    }
    if (this->m_includingSingleFT) {
        m_single_FD = new std::complex<T>[this->m_settingsInv->fDim().xyzPadded()];
        m_singleToFD = FFTwT<T>::createPlan(*m_cpu, settingsForw.createSingle());
    }
}

template<typename T>
void ShiftCorrEstimator<T>::release() {
    AShiftCorrEstimator<T>::release();

    // host memory
    if (this->m_includingSingleFT) {
        delete[] m_single_FD;
    }
    if (this->m_includingBatchFT) {
        delete[] m_batch_FD;
    }

    // FT plans
    FFTwT<T>::release(m_singleToFD);
    FFTwT<T>::release(m_batchToFD);
    FFTwT<T>::release(m_batchToSD);

    setDefault();
}

template<typename T>
void ShiftCorrEstimator<T>::setDefault() {
    AShiftCorrEstimator<T>::setDefault();

    m_cpu = nullptr;

    // host memory
    m_single_FD = nullptr;
    m_batch_FD = nullptr;

    // plans
    m_singleToFD = nullptr;
    m_batchToFD = nullptr;
    m_batchToSD = nullptr;
}


template<typename T>
void ShiftCorrEstimator<T>::load2DReferenceOneToN(const std::complex<T> *ref) {
    auto isReady = (this->m_isInit && (AlignType::OneToN == this->m_type));
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to load a reference signal");
    }

    // simply remember the pointer. Expect that nobody will change it meanwhile
    // We won't change it, but since generally speaking, we do change this pointer
    // remove the const
    m_single_FD = const_cast<std::complex<T>*>(ref);

    // update state
    this->m_is_ref_FD_loaded = true;
}

template<typename T>
void ShiftCorrEstimator<T>::computeCorrelations2DOneToN(
        std::complex<T> *inOut, bool center) {
    bool isReady = (this->m_isInit && (AlignType::OneToN == this->m_type) && this->m_is_ref_FD_loaded);

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() before");
    }

    sComputeCorrelations2DOneToN(
        *m_cpu,
        inOut, m_single_FD,
        this->m_settingsInv->fDim().x(),
        this->m_settingsInv->fDim().y(),
        this->m_settingsInv->fDim().n(),
        center);
}

template<typename T>
void ShiftCorrEstimator<T>::computeCorrelations2DOneToN(
        const HW &hw,
        std::complex<T> *inOut,
        const std::complex<T> *ref,
        size_t xDim, size_t yDim, size_t nDim, bool center) {
    const CPU *cpu;
    try {
        cpu = &dynamic_cast<const CPU&>(hw);
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of CPU expected");
    }
    return sComputeCorrelations2DOneToN(*cpu, inOut, ref, xDim, yDim, nDim, center);
}

template<typename T>
void ShiftCorrEstimator<T>::sComputeCorrelations2DOneToN(
        const HW &hw,
        std::complex<T> *inOut,
        const std::complex<T> *ref,
        size_t xDim, size_t yDim, size_t nDim, bool center) {
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
template class ShiftCorrEstimator<float>;
template class ShiftCorrEstimator<double>;

} /* namespace Alignment */
