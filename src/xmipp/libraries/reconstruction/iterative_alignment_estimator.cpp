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
T *IterativeAlignmentEstimator<T>::applyTransform(const AlignmentEstimation &estimation) {
    std::vector<float> t;
    t.reserve(9 * estimation.poses.size());
    auto tmp = Matrix2D<float>(3, 3);
    SPEED_UP_temps0
    for (size_t j = 0; j < estimation.poses.size(); ++j) {
        M3x3_INV(tmp, estimation.poses.at(j)) // inverse the transformation
        for (int i = 0; i < 9; ++i) {
            t.emplace_back(tmp.mdata[i]);
        }
    }
    return m_transformer.interpolate(t);
}

template<typename T>
void IterativeAlignmentEstimator<T>::sApplyTransform(ctpl::thread_pool &pool, const Dimensions &dims, // FIXME DS remove
        const AlignmentEstimation &estimation,
        const T * __restrict__ orig, T * __restrict__ copy, bool hasSingleOrig) {
    const size_t n = dims.n();
    const size_t z = dims.z();
    const size_t y = dims.y();
    const size_t x = dims.x();

    auto futures = std::vector<std::future<void>>();

    auto workload = [&](int id, size_t signalId){
        size_t offset = signalId * dims.sizeSingle();
        auto in = MultidimArray<T>(1, z, y, x, const_cast<T*>(orig + (hasSingleOrig ? 0 : offset))); // removing const, but data should not be changed
        auto out = MultidimArray<T>(1, z, y, x, copy + offset);
        in.setXmippOrigin();
        out.setXmippOrigin();
        // compensate the movement
        applyGeometry(LINEAR, out, in, estimation.poses.at(signalId), false, DONT_WRAP);
    };

    for (size_t i = 0; i < n; ++i) {
        futures.emplace_back(pool.push(workload, i));
    }
    for (auto &f : futures) {
        f.get();
    }
}

template<typename T>
void IterativeAlignmentEstimator<T>::compute(unsigned iters, AlignmentEstimation &est,
        bool rotationFirst) {
    // note (DS) if any of these steps return 0 (no shift or rotation), additional iterations are useless
    // as the image won't change
    auto stepRotation = [&] {
        m_rot_est.compute(m_transformer.getDest());
        const auto &cRotEst = m_rot_est;
        auto r = Matrix2D<float>();
        updateEstimation(est,
            cRotEst.getRotations2D(),
            [&r](float angle, Matrix2D<float> &lhs) {
                rotation2DMatrix(angle, r);
                lhs = r * lhs;
            });
        applyTransform(est);
    };
    auto stepShift = [&] {
        m_shift_est.computeShift2DOneToN(m_transformer.getDest());
        updateEstimation(est, m_shift_est.getShifts2D(),
                [](const Point2D<float> &shift, Matrix2D<float> &lhs) {
            MAT_ELEM(lhs, 0, 2) += shift.x;
            MAT_ELEM(lhs, 1, 2) += shift.y;
        });
        applyTransform(est);
    };
    // get a fresh copy of the images
    m_transformer.copySrcToDest();
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
    m_meritComputer.compute(m_transformer.getDest());
    const auto &mc = m_meritComputer;
    est.figuresOfMerit = mc.getFiguresOfMerit();
}

template<typename T>
void IterativeAlignmentEstimator<T>::loadReference(
        const T *ref) {
    m_meritComputer.loadReference(ref);
    m_shift_est.load2DReferenceOneToN(ref);
    if ( ! m_sameEstimators) {
        m_rot_est.loadReference(ref);
    }
}

template<typename T>
AlignmentEstimation IterativeAlignmentEstimator<T>::compute(
        const T * __restrict__ others, // it would be good if data is normalized, but probably it does not have to be
        unsigned iters) {
    m_transformer.setSrc(others);

    // prepare transformer which is responsible for applying the pose t
    const size_t n = m_rot_est.getSettings().otherDims.n();
    // try rotation -> shift
    auto result_RS = AlignmentEstimation(n);
    compute(iters, result_RS, true);
    // try shift-> rotation
    auto result_SR = AlignmentEstimation(n);
    compute(iters, result_SR, false);

    for (size_t i = 0; i < n; ++i) {
        if (result_RS.figuresOfMerit.at(i) < result_SR.figuresOfMerit.at(i)) {
            result_RS.figuresOfMerit.at(i) = result_SR.figuresOfMerit.at(i);
            result_RS.poses.at(i) = result_SR.poses.at(i);
        }
    }

    return result_RS;
}

template<typename T>
void IterativeAlignmentEstimator<T>::check() {
    if (m_rot_est.getSettings().otherDims != m_shift_est.getDimensions()) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Estimators are initialized for different sizes");
    }
    if (m_rot_est.getSettings().type != m_shift_est.getAlignType()) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Estimators are initialized for different type of alignment");
    }
}

// explicit instantiation
template class IterativeAlignmentEstimator<float>;
template class IterativeAlignmentEstimator<double>;

} /* namespace Alignment */
