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

namespace Alignment {

template<typename T>
class IterativeAlignmentEstimator {
public:
    IterativeAlignmentEstimator(ARotationEstimator<T> &rot_estimator,
            AShiftEstimator<T> &shift_estimator) :
                m_rot_est(rot_estimator), m_shift_est(shift_estimator),
                m_dims(shift_estimator.getDimensions()) {
        m_sameEstimators = ((void*)&m_shift_est == (void*)&m_rot_est);
        this->check();
    }

    AlignmentEstimation compute(__restrict const T *ref, __restrict const T *others,
            unsigned iters = 3);


protected:
    static void sApplyTransform(const Dimensions &dims,
                const AlignmentEstimation &estimation,
                __restrict const T *orig, __restrict T *copy);

private:
    ARotationEstimator<T> &m_rot_est;
    AShiftEstimator<T> &m_shift_est;
    const Dimensions &m_dims;
    bool m_sameEstimators;

    template<typename U, typename F>
    void updateEstimation(AlignmentEstimation &est,
            const U &newVals, const F &func);

    void compute(unsigned iters, AlignmentEstimation &est,
            __restrict const T *ref,
            __restrict const T *orig,
            __restrict T *copy,
            bool rotationFirst);

    void computeCorrelation(AlignmentEstimation &estimation,
            __restrict const T *orig, __restrict T *copy);

    void check();

    void print(const AlignmentEstimation &e);
};


} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_ITERATIVE_ALIGNMENT_ESTIMATOR_H_ */
