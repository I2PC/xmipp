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
        const T * __restrict__ orig, T * __restrict__ copy, bool hasSingleOrig) {
    static size_t counter = 0;
    const size_t n = dims.n();
    const size_t z = dims.z();
    const size_t y = dims.y();
    const size_t x = dims.x();

    int threads = 4;

    auto workers = std::vector<std::thread>();
    int imgsPerWorker = std::ceil(n / (float)threads);

    auto workload = [&](int id){
        const size_t first = id * imgsPerWorker;
        const size_t last = std::min(first + imgsPerWorker, n);
        for (size_t i = first; i < last; ++i) {
            size_t offset = i * dims.sizeSingle();
            auto in = MultidimArray<T>(1, z, y, x, const_cast<T*>(orig + (hasSingleOrig ? 0 : offset))); // removing const, but data should not be changed
            auto out = MultidimArray<T>(1, z, y, x, copy + offset);
            in.setXmippOrigin();
            out.setXmippOrigin();
            // compensate the movement
            applyGeometry(LINEAR, out, in, estimation.poses.at(i), false, DONT_WRAP);
        }
    };

    for (size_t w = 0; w < threads; ++w) {
        workers.emplace_back(workload, w);
    }
    for (auto &w : workers) {
        w.join();
    }
}

template<typename T>
void IterativeAlignmentEstimator<T>::computeCorrelation(AlignmentEstimation &estimation,
        const T * __restrict__ orig, T * __restrict__ copy) {
    const size_t n = m_dims.n();
    const size_t z = m_dims.z();
    const size_t y = m_dims.y();
    const size_t x = m_dims.x();

    int threads = 4;

    auto workers = std::vector<std::thread>();
    int imgsPerWorker = std::ceil(n / (float)threads);

    auto workload = [&](int id){
        const size_t first = id * imgsPerWorker;
        const size_t last = std::min(first + imgsPerWorker, n);
        for (size_t i = first; i < last; ++i) {
            T * address = copy + i * m_dims.sizeSingle();
            auto ref = MultidimArray<T>(1, z, y, x, const_cast<T*>(orig)); // removing const, but data should not be changed
            auto other = MultidimArray<T>(1, z, y, x, address);
            // FIXME DS better if we use fastCorrelation, but unless the input is normalized
            // we won't receive correlation in [0..1], so we won't be able to directly compare
            // against the original version of the algorithm
            estimation.correlations.at(i) = correlationIndex(ref, other);
        }
    };

    for (size_t w = 0; w < threads; ++w) {
        workers.emplace_back(workload, w);
    }
    for (auto &w : workers) {
        w.join();
    }
}

template<typename T>
void IterativeAlignmentEstimator<T>::compute(unsigned iters, AlignmentEstimation &est,
        const T * __restrict__ ref,
        const T * __restrict__ orig,
        T * __restrict__ copy,
        bool rotationFirst) {
    auto stepRotation = [&] {
        m_rot_est.compute(copy);
        const auto &cRotEst = m_rot_est;
        updateEstimation(est,
            cRotEst.getRotations2D(),
            [](float angle, Matrix2D<double> &lhs) {
                auto r = Matrix2D<double>();
                rotation2DMatrix(angle, r);
                lhs = r * lhs;
            });
        sApplyTransform(m_dims, est, orig, copy, false);
    };
    auto stepShift = [&] {
        m_shift_est.computeShift2DOneToN(copy);
        updateEstimation(est, m_shift_est.getShifts2D(),
                [](const Point2D<float> &shift, Matrix2D<double> &lhs) {
            MAT_ELEM(lhs, 0, 2) += shift.x;
            MAT_ELEM(lhs, 1, 2) += shift.y;
        });
        sApplyTransform(m_dims, est, orig, copy, false);
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
        const T *__restrict__ ref, const T * __restrict__ others, // it would be good if data is normalized, but probably it does not have to be
        unsigned iters) {

    m_shift_est.load2DReferenceOneToN(ref);
    if ( ! m_sameEstimators) {
        m_rot_est.loadReference(ref);
    }

    // allocate memory for signals with applied pose
    size_t elems = m_dims.sizePadded();
    auto copy = memoryUtils::page_aligned_alloc<T>(elems, false);
    m_shift_est.getHW().lockMemory(copy, elems * sizeof(T));
    m_rot_est.getHW().lockMemory(copy, elems * sizeof(T));

    const size_t n = m_dims.n();
    // try rotation -> shift
    auto result_RS = AlignmentEstimation(n);
    memcpy(copy, others, elems * sizeof(T));
    compute(iters, result_RS, ref, others, copy, true);
    // try shift-> rotation
    auto result_SR = AlignmentEstimation(n);
    memcpy(copy, others, elems * sizeof(T));
    compute(iters, result_SR, ref, others, copy, false);

    m_rot_est.getHW().unlockMemory(copy);
    m_shift_est.getHW().unlockMemory(copy);
    free(copy);

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
