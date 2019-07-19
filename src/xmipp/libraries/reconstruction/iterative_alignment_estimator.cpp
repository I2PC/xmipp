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

#include "iterative_alignment_estimator.h"

namespace Alignment {

template<typename T>
template<typename U, typename F>
void IterativeAlignmentEstimator<T>::updateEstimation(AlignmentEstimation &est,
        const U &newVals, const F &func) {
    size_t len = newVals.size();
    for (size_t i = 0; i < len; ++i) {
        func(newVals.at(i), est.poses.at(i));
    }
}

template<typename T>
void IterativeAlignmentEstimator<T>::print(const AlignmentEstimation &e) {
    for (auto &m : e.poses) {
        printf("([%f %f], %f) ", MAT_ELEM(m, 0, 2), MAT_ELEM(m, 1, 2), RAD2DEG(atan2(MAT_ELEM(m, 1, 0), MAT_ELEM(m, 0, 0))));
    }
    printf("\n");
}

template<typename T>
void IterativeAlignmentEstimator<T>::sApplyTransform(const Dimensions &dims,
        const AlignmentEstimation &estimation,
        __restrict const T *orig, __restrict T *copy) {
    static size_t counter = 0;
    const size_t n = dims.n();
    const size_t z = dims.z();
    const size_t y = dims.y();
    const size_t x = dims.x();
    for (size_t i = 0; i < n; ++i) {
        auto in = MultidimArray<T>(1, z, y, x, const_cast<T*>(orig)); // removing const, but data should not be changed
        auto out = MultidimArray<T>(1, z, y, x, copy);
        in.setXmippOrigin();
        out.setXmippOrigin();
        // compensate the movement
        applyGeometry(LINEAR, out, in, estimation.poses.at(i), false, DONT_WRAP);
        // move pointers
        orig += dims.xyzPadded();
        copy += dims.xyzPadded();
    }

}

template<typename T>
void IterativeAlignmentEstimator<T>::computeCorrelation(AlignmentEstimation &estimation,
        __restrict const T *orig, __restrict T *copy) {
    const size_t n = m_dims.n();
    const size_t z = m_dims.z();
    const size_t y = m_dims.y();
    const size_t x = m_dims.x();
    MultidimArray<T> ref;
    MultidimArray<T> other;
    for (size_t i = 0; i < n; ++i) {
        ref = MultidimArray<T>(1, z, y, x, const_cast<T*>(orig)); // removing const, but data should not be changed
        other = MultidimArray<T>(1, z, y, x, copy);
        estimation.correlations.at(i) = fastCorrelation(ref, other);
    }
}

template<typename T>
void IterativeAlignmentEstimator<T>::compute(unsigned iters, AlignmentEstimation &est,
        __restrict const T *ref,
        __restrict const T *orig,
        __restrict T *copy,
        bool rotationFirst) {
    auto stepRotation = [&] {
        m_rot_est.compute(copy);
        updateEstimation(est, m_rot_est.getRotations2D(), [](float angle, Matrix2D<double> &lhs) {
            auto r = Matrix2D<double>();
            rotation2DMatrix(angle, r);
            lhs = r * lhs;
        });
        sApplyTransform(m_dims, est, orig, copy);
    };
    auto stepShift = [&] {
        m_shift_est.computeShift2DOneToN(copy);
        updateEstimation(est, m_shift_est.getShifts2D(),
                [](const Point2D<float> &shift, Matrix2D<double> &lhs) {
            MAT_ELEM(lhs, 0, 2) += shift.x;
            MAT_ELEM(lhs, 1, 2) += shift.y;
        });
        sApplyTransform(m_dims, est, orig, copy);
    };
    for (unsigned i = 0; i < iters; ++i) {
        if (rotationFirst) {
            stepRotation();
            // if we have object implementing both interfaces, we don't need to run next step
            if ( ! m_sameEstimators) {
                stepShift();
            }
        } else {
            stepShift();
            // if we have object implementing both interfaces, we don't need to run next step
            if ( ! m_sameEstimators) {
                stepRotation();
            }
        }
//        print(est);
    }
    computeCorrelation(est, ref, copy);
}

template<typename T>
AlignmentEstimation IterativeAlignmentEstimator<T>::compute(
        __restrict const T *ref, __restrict const T *others,
        unsigned iters) {
    m_shift_est.load2DReferenceOneToN(ref);
    if ( ! m_sameEstimators) {
        m_rot_est.loadReference(ref);
    }

    // allocate memory for signals with applied pose
    size_t elems = m_dims.sizePadded();
    auto copy = new T[elems];

    const size_t n = m_dims.n();
    // try rotation -> shift
    auto result_RS = AlignmentEstimation(n);
    memcpy(copy, others, elems * sizeof(T));
    compute(iters, result_RS, ref, others, copy, true);
    // try shift-> rotation
    auto result_SR = AlignmentEstimation(n);
    memcpy(copy, others, elems * sizeof(T));
    compute(iters, result_SR, ref, others, copy, false);

    delete[] copy;

    for (size_t i = 0; i < n; ++i) {
        if (result_RS.correlations.at(i) < result_SR.correlations.at(i)) {
            result_RS.correlations.at(i) = result_SR.correlations.at(i);
            result_RS.poses.at(i) = result_SR.poses.at(i);
        }
    }
    return result_RS;
}

template<typename T>
void IterativeAlignmentEstimator<T>::check() {
    if ( ! (m_rot_est.isInitialized() && m_shift_est.isInitialized())) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Estimators are not initialized");
    }
    if (m_rot_est.getDimensions() != m_shift_est.getDimensions()) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Estimators are initialized for different sizes");
    }
    if (m_rot_est.getAlignType() != m_shift_est.getAlignType()) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Estimators are initialized for different type of alignment");
    }
}

// explicit instantiation
template class IterativeAlignmentEstimator<float>;
template class IterativeAlignmentEstimator<double>;

} /* namespace Alignment */
