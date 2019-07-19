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
void PolarRotationEstimator<T>::init2D(const HW &hw) {
    try {
        m_cpu = &dynamic_cast<const CPU&>(hw);
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of CPU expected");
    }

    m_firstRing = this->m_dims->x() / 5;
    m_lastRing = this->m_dims->x() / 2;
    if (std::is_same<T, float>()) {
        m_dataAux.resize(this->m_dims->y(), this->m_dims->x());
    }

    this->m_isInit = true;
}

template<typename T>
void PolarRotationEstimator<T>::load2DReferenceOneToN(const T *ref) {
    MultidimArray<double> tmp = convert(const_cast<T*>(ref));
    tmp.setXmippOrigin();
    normalizedPolarFourierTransform(tmp, m_refPolarFourierI, false,
            m_firstRing, m_lastRing, m_refPlans, 1);
    m_rotCorrAux.resize(2 * m_refPolarFourierI.getSampleNoOuterRing() - 1);
    m_aux.local_transformer.setReal(m_rotCorrAux);
    this->m_is_ref_loaded = true;
}

template<>
MultidimArray<double> PolarRotationEstimator<float>::convert(float *data) {
    const size_t s = this->m_dims->xyz();
    for (size_t i = 0; i < s; ++i) {
        m_dataAux.data[i] = data[i];
    }
    return m_dataAux;
}

template<>
MultidimArray<double> PolarRotationEstimator<double>::convert(double *data) {
    return MultidimArray<double>(
            this->m_dims->n(), this->m_dims->z(),
            this->m_dims->y(), this->m_dims->x(),
            data);
}

template<typename T>
void PolarRotationEstimator<T>::computeRotation2DOneToN(T *others) {
    bool isReady = (this->m_isInit && (AlignType::OneToN == this->m_type) && this->m_is_ref_loaded);

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() and load reference");
    }

    this->m_rotations2D.reserve(this->m_dims->n());
    for (size_t n = 0; n < this->m_dims->n(); ++n) {
        size_t offset = n * this->m_dims->xyzPadded();
        MultidimArray<double> tmp = convert(others + offset);
        tmp.setXmippOrigin();
        normalizedPolarFourierTransform(tmp, m_polarFourierI, true,
                m_firstRing, m_lastRing, m_plans, 1);
        this->m_rotations2D.emplace_back(
                best_rotation(m_refPolarFourierI, m_polarFourierI, m_aux));
    }
    this->m_is_rotation_computed = true;
}

template<typename T>
void PolarRotationEstimator<T>::release() {
    delete m_plans;
    delete m_refPlans;
    m_rotCorrAux.clear();
    m_dataAux.clear();
    ARotationEstimator<T>::release();
    PolarRotationEstimator<T>::setDefault();
}

template<typename T>
void PolarRotationEstimator<T>::setDefault() {
    m_plans = nullptr;
    m_refPlans = nullptr;
    m_polarFourierI = Polar<std::complex<double>>();
    m_refPolarFourierI = Polar<std::complex<double>>();
    m_firstRing = -1;
    m_lastRing = -1;
    ARotationEstimator<T>::setDefault();
}

template<typename T>
void PolarRotationEstimator<T>::check() {
    ARotationEstimator<T>::check();
    if (this->m_batch != 1) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "This estimator cannot work with batched signals");
    }
    if (this->m_dims->x() != this->m_dims->y()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "This estimator can work only with square signal");
    }
}

// explicit instantiation
template class PolarRotationEstimator<float>;
template class PolarRotationEstimator<double>;

} /* namespace Alignment */
