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

#ifndef LIBRARIES_RECONSTRUCTION_ITERATIVE_ALIGNMENT_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_ITERATIVE_ALIGNMENT_ESTIMATOR_H_

#include "arotation_estimator.h"
#include "ashift_estimator.h"
#include "align_type.h"
#include "data/alignment_estimation.h"
#include "core/transformations.h"
#include "data/filters.h"
#include <core/utils/memory_utils.h>
#include "CTPL/ctpl_stl.h"
#include "reconstruction/bspline_geo_transformer.h"
#include "amerit_computer.h"

namespace Alignment {

template<typename T>
class IterativeAlignmentEstimator {
public:
    IterativeAlignmentEstimator(ARotationEstimator<T> &rot_estimator,
            AShiftEstimator<T> &shift_estimator,
            BSplineGeoTransformer<T> &interpolator,
            AMeritComputer<T> &meritComputer,
            ctpl::thread_pool &threadPool) :
                m_rot_est(rot_estimator),
                m_shift_est(shift_estimator),
                m_meritComputer(meritComputer),
                m_threadPool(threadPool),
                m_transformer(interpolator) {
        m_sameEstimators = ((void*)&m_shift_est == (void*)&m_rot_est);
        this->check();
    }

    void loadReference(const T *ref);

    AlignmentEstimation compute(const T *others, // it would be good if data is normalized, but probably it does not have to be
            unsigned iters = 3);
protected:
    static void sApplyTransform(ctpl::thread_pool &pool, const Dimensions &dims, // FIXME DS remove, also includes
                const AlignmentEstimation &estimation,
                const T *orig, T *copy, bool hasSingleOrig);

private:
    ARotationEstimator<T> &m_rot_est;
    AShiftEstimator<T> &m_shift_est;
    BSplineGeoTransformer<T> &m_transformer;
    AMeritComputer<T> &m_meritComputer;
    ctpl::thread_pool &m_threadPool;
    bool m_sameEstimators;

    T *applyTransform(const AlignmentEstimation &estimation);

    template<typename U, typename F>
    void updateEstimation(AlignmentEstimation &est,
            const U &newVals, const F &func);

    void compute(unsigned iters, AlignmentEstimation &est,
            bool rotationFirst);

    void check();

    void print(const AlignmentEstimation &e);
};


} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_ITERATIVE_ALIGNMENT_ESTIMATOR_H_ */
