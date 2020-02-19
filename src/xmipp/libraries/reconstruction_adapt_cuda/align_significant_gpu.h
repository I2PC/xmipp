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

#ifndef LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_ALIGN_SIGNIFICANT_GPU_H_
#define LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_ALIGN_SIGNIFICANT_GPU_H_

#include "reconstruction/aalign_significant.h"
#include "reconstruction/iterative_alignment_estimator.h"
#include "reconstruction_cuda/cuda_rot_polar_estimator.h"
#include "reconstruction_cuda/cuda_shift_corr_estimator.h"
#include "reconstruction_cuda/cuda_bspline_geo_transformer.h"
#include "reconstruction_cuda/cuda_correlation_computer.h"

#include <algorithm>
#include <numeric> // std::accumulate

namespace Alignment {

template<typename T>
class ProgAlignSignificantGPU : public AProgAlignSignificant<T> {
using typename AProgAlignSignificant<T>::Assignment;
protected:
    std::vector<AlignmentEstimation> align(const T *ref, const T *others) override;

    void updateRefs(T *refs, const T *others,
            const std::vector<Assignment> &assignments) override;

private:
    void initRotEstimator(CudaRotPolarEstimator<T> &est, std::vector<HW*> &hw, const Dimensions &dims);
    void initShiftEstimator(CudaShiftCorrEstimator<T> &est, std::vector<HW*> &hw, const Dimensions &dims);
    void initTransformer(BSplineGeoTransformer<T> &t, std::vector<HW*> &hw, const Dimensions &dims);
    void initMeritComputer(AMeritComputer<T> &mc, std::vector<HW*> &hw, const Dimensions &dims);

    void interpolate(BSplineGeoTransformer<T> &transformer,
            T *data,
            const std::vector<Assignment> &assignments,
            std::vector<float> &matrices,
            size_t offset,
            size_t toProcess);

    size_t m_maxBatchSize = 300;
};


} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_ALIGN_SIGNIFICANT_GPU_H_ */
