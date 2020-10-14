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

#include "arotation_estimator.h"

namespace Alignment {

template<typename T>
void ARotationEstimator<T>::init(const RotationEstimationSetting settings, bool reuse) {
    // check that settings is not completely wrong
    settings.check();
    bool skipInit = m_isInit && reuse && this->canBeReused(settings);
    // set it
    m_settings = settings;
    // initialize estimator
    if ( ! skipInit) {
        if (m_settings.otherDims.is2D()) {
            this->init2D();
        } else {
            REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
        }
        // check that there's no logical problem
        this->check();
        // no issue found, we're good to go
        m_isInit = true;
    }
}

template<typename T>
bool ARotationEstimator<T>::canBeReused(const RotationEstimationSetting &s) const {
    if (m_settings.otherDims.is2D()) {
        return this->canBeReused2D(s);
    }
    REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
}

template<typename T>
void ARotationEstimator<T>::loadReference(const T *ref) {
    if (m_settings.otherDims.is2D()) {
        if (AlignType::OneToN == m_settings.type) {
            this->load2DReferenceOneToN(ref);
            m_isRefLoaded = true;
            return;
        }
    }
    REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
}

template<typename T>
void ARotationEstimator<T>::compute(T *others) { // FIXME DS should be const?
    if (m_settings.otherDims.is2D()) {
        m_rotations2D.resize(0);
        m_rotations2D.reserve(m_settings.otherDims.n());
        if (AlignType::OneToN == m_settings.type) {
            return this->computeRotation2DOneToN(others);
        }
    }
    REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
}

// explicit instantiation
template class ARotationEstimator<float>;
template class ARotationEstimator<double>;

} /* namespace Alignment */
