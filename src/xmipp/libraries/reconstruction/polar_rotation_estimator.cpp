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

#include "polar_rotation_estimator.h"

namespace Alignment {

template<typename T>
void PolarRotationEstimator<T>::init2D() {
    release();
    if (1 != this->getSettings().hw.size()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Only one CPU thread expected");
    }
    try {
        m_cpu = dynamic_cast<CPU*>(this->getSettings().hw.at(0));
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of CPU expected");
    }

    auto dims = this->getSettings().refDims;

    m_firstRing = dims.x() / 5;
    m_lastRing = (dims.x() - 3) / 2;  // so that we have some edge around the biggest ring
    // all rings have the same number of samples, to make FT easier
    if (std::is_same<T, float>()) {
        m_dataAux.resize(dims.y(), dims.x());
    }
}

template<typename T>
void PolarRotationEstimator<T>::load2DReferenceOneToN(const T *ref) {
    auto isReady = this->isInitialized();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to load a reference signal");
    }

    MultidimArray<double> tmp = convert(const_cast<T*>(ref));
    tmp.setXmippOrigin();
    normalizedPolarFourierTransform(tmp, m_refPolarFourierI, false,
            m_firstRing, m_lastRing, m_refPlans, 1);
    m_rotCorrAux.resize(2 * m_refPolarFourierI.getSampleNoOuterRing() - 1);
    m_aux.local_transformer.setReal(m_rotCorrAux);
}

template<>
MultidimArray<double> PolarRotationEstimator<float>::convert(float *data) {
    const size_t s = this->getSettings().otherDims.sizeSingle();
    for (size_t i = 0; i < s; ++i) {
        m_dataAux.data[i] = data[i];
    }
    return m_dataAux;
}

template<>
MultidimArray<double> PolarRotationEstimator<double>::convert(double *data) {
    const auto s = this->getSettings().otherDims.copyForN(1);
    return MultidimArray<double>(
            s.n(), s.z(),
            s.y(), s.x(),
            data);
}

template<typename T>
void PolarRotationEstimator<T>::computeRotation2DOneToN(T *others) {
    bool isReady = this->isInitialized() && this->isRefLoaded();

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() and load reference");
    }

    const auto dims = this->getSettings().otherDims;
    for (size_t n = 0; n < dims.n(); ++n) {
        size_t offset = n * dims.sizeSingle();
        MultidimArray<double> tmp = convert(others + offset);
        tmp.setXmippOrigin();
        normalizedPolarFourierTransform(tmp, m_polarFourierI, true,
                m_firstRing, m_lastRing, m_plans, 1);
        this->getRotations2D().emplace_back(
                best_rotation(m_refPolarFourierI, m_polarFourierI, m_aux));
    }
}

template<typename T>
void PolarRotationEstimator<T>::release() {
    delete m_plans;
    delete m_refPlans;
    m_rotCorrAux.clear();
    m_dataAux.clear();

    setDefault();
}

template<typename T>
void PolarRotationEstimator<T>::setDefault() {
    m_plans = nullptr;
    m_refPlans = nullptr;
    m_polarFourierI = Polar<std::complex<double>>();
    m_refPolarFourierI = Polar<std::complex<double>>();
    m_firstRing = -1;
    m_lastRing = -1;
}

template<typename T>
void PolarRotationEstimator<T>::check() {
    const auto s = this->getSettings();
    if (s.batch != 1) {
        std::cerr << "Batch processing is not supported. Signals will be processed one by one.\n";
    }
    if (s.refDims.x() != s.refDims.y()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "This estimator can work only with square signal");
    }
    if (s.refDims.isPadded()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Padded signal is not supported");
    }
    if (s.refDims.x() < 6) {
        // we need some edge around the biggest ring, to avoid invalid memory access
        REPORT_ERROR(ERR_ARG_INCORRECT, "The input signal is too small.");
    }
}

// explicit instantiation
template class PolarRotationEstimator<float>;
template class PolarRotationEstimator<double>;

} /* namespace Alignment */
