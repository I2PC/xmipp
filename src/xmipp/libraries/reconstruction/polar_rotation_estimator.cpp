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
    auto s = this->getSettings();
    if (1 != s.hw.size()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Only one CPU thread expected");
    }
    try {
        m_cpu = dynamic_cast<CPU*>(s.hw.at(0));
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of CPU expected");
    }

    if (std::is_same<T, float>()) {
        m_dataAux.resize(s.refDims.y(), s.refDims.x());
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
    auto s = this->getSettings();
    polarFourierTransform<false>(tmp, m_refPolarFourierI, false,
            s.firstRing, s.lastRing, m_refPlans, 1);
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
    auto s = this->getSettings();
    for (size_t n = 0; n < s.otherDims.n(); ++n) {
        size_t offset = n * s.otherDims.sizeSingle();
        MultidimArray<double> tmp = convert(others + offset);
        tmp.setXmippOrigin();
        polarFourierTransform<false>(tmp, m_polarFourierI, true,
                s.firstRing, s.lastRing, m_plans, 1);
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
}

template<typename T>
void PolarRotationEstimator<T>::check() {
    const auto s = this->getSettings();
    if (s.batch != 1) {
        std::cerr << "Batch processing is not supported. Signals will be processed one by one.\n";
    }
    if (s.allowDataOverwrite) {
        std::cerr << "allowDataOverwrite flag is ignored, as it's not yet supported.\n";
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
