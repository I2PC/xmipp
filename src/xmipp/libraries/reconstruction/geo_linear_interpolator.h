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

#ifndef LIBRARIES_RECONSTRUCTION_GEO_LINEAR_INTERPOLATOR_H_
#define LIBRARIES_RECONSTRUCTION_GEO_LINEAR_INTERPOLATOR_H_

#include "reconstruction/ageo_linear_interpolator.h"
#include "data/dimensions.h"
#include "data/cpu.h"
#include <CTPL/ctpl_stl.h>
#include "data/filters.h"
#include <core/utils/memory_utils.h>

//FIXME DS rework properly

template<typename T>
class GeoLinearTransformer : public AGeoLinearTransformer<T> {
public:
    GeoLinearTransformer(Dimensions d) :
        dims(d) {
        m_threadPool.resize(CPU::findCores());
        m_dest = memoryUtils::page_aligned_alloc<T>(d.size(), false);
        m_src = nullptr;
    };

    virtual ~GeoLinearTransformer() {delete[] m_dest;};

    void createCopyOnGPU(const T *h_data) override {m_src = h_data;};

    T *interpolate(const std::vector<float> &matrices) override; // each 3x3 values are a single matrix
private:
    Dimensions dims;
    const T *m_src;
    T *m_dest;
    ctpl::thread_pool m_threadPool;
};


#endif /* LIBRARIES_RECONSTRUCTION_GEO_LINEAR_INTERPOLATOR_H_ */
