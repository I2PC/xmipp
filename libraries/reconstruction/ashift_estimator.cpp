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

#include "ashift_estimator.h"

namespace Alignment {

template<typename T>
void AShiftEstimator<T>::release() {
    delete m_dims;

    m_shifts2D.clear();

    AShiftEstimator<T>::setDefault();
}

template<typename T>
void AShiftEstimator<T>::setDefault() {
    m_type = AlignType::None;
    m_dims = nullptr;
    m_batch = 0;
    m_maxShift = Point3D<size_t>(0, 0, 0);

    m_shifts2D.reserve(0);

    m_isInit = false;
    m_is_ref_loaded = false;
    m_is_shift_computed = false;
}

template<typename T>
void AShiftEstimator<T>::init2D(AlignType type, const Dimensions &dims,
               size_t batch, Point2D<size_t> maxShift) {
    m_type = type;
    m_dims = new Dimensions(dims);
    m_batch = std::min(batch, m_dims->n());
    m_maxShift = Point3D<size_t>(maxShift.x, maxShift.y, 1);

    AShiftEstimator<T>::check();
}

template<typename T>
void AShiftEstimator<T>::check() {
    if ((0 == m_dims->x()) || (0 == m_dims->y())
            || (0 == m_dims->z()) || (0 == m_dims->n())) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "One of the dimensions is zero (0)");
    }
    if (0 == m_batch) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "Batch is zero (0)");
    }
    if ((0 == m_maxShift.x) || (0 == m_maxShift.y)
        || (0 == m_maxShift.z)) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "Max shift in some dimension is zero (0)");
    }
    if ((m_maxShift.x > m_dims->x()) || (m_maxShift.y > m_dims->y())
        || (m_maxShift.z > m_dims->z())) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "Max shift in some dimension is bigger than"
                "the dimension");
    }
}

// explicit instantiation
template class AShiftEstimator<float>;
template class AShiftEstimator<double>;

}
