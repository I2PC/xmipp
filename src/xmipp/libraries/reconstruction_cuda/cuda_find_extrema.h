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

#ifndef LIBRARIES_RECONSTRUCTION_CUDA_FIND_EXTREMA_H_
#define LIBRARIES_RECONSTRUCTION_CUDA_FIND_EXTREMA_H_

#include "reconstruction/afind_extrema.h"
#include "reconstruction_cuda/gpu.h"
#include <limits>

namespace ExtremaFinder {

template<typename T>
class CudaExtremaFinder : public AExtremaFinder<T> {
public:
    CudaExtremaFinder() {
        setDefault();
    }

    virtual ~CudaExtremaFinder() {
        release();
    }

    CudaExtremaFinder(CudaExtremaFinder& o) = delete;
    CudaExtremaFinder& operator=(const CudaExtremaFinder& other) = delete;
    CudaExtremaFinder const & operator=(CudaExtremaFinder &&o) = delete;
    CudaExtremaFinder(CudaExtremaFinder &&o); // FIXME DS implement

    static void sFindMax(const GPU &gpu,
        const Dimensions &dims,
        const T * __restrict__ d_data,
        T * __restrict__ d_positions,
        T * __restrict__ d_values);

    static void sFindMax2DNearCenter(const GPU &gpu,
        const Dimensions &dims,
        const T * data,
        T * positions,
        T * values,
        size_t maxDist);

    static size_t ceilPow2(size_t x); // FIXME DS move this to somewhere else

private:
    GPU *m_loadStream;
    GPU *m_workStream;

    void check() override;
    void setDefault();
    void release();

    void initMax(bool reuse) override;
    void findMax(T *data) override;

    void initMaxAroundCenter(bool reuse) override;
    void findMaxAroundCenter(T *data) override;
};

} /* namespace ExtremaFinder */

#endif /* LIBRARIES_RECONSTRUCTION_CUDA_FIND_EXTREMA_H_ */
