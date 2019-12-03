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

#ifndef LIBRARIES_RECONSTRUCTION_AGEO_LINEAR_INTERPOLATOR_H_
#define LIBRARIES_RECONSTRUCTION_AGEO_LINEAR_INTERPOLATOR_H_

#include "data/dimensions.h"
#include <vector>

//FIXME DS rework properly

template<typename T>
class AGeoLinearTransformer {
public:
    virtual ~AGeoLinearTransformer() {};
    virtual void createCopyOnGPU(const T *h_data)  = 0;

    virtual T *interpolate(const std::vector<float> &matrices)  = 0; // each 3x3 values are a single matrix
};


#endif /* LIBRARIES_RECONSTRUCTION_AGEO_LINEAR_INTERPOLATOR_H_ */
