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

#ifndef LIBRARIES_DATA_LOCAL_ALIGNMENT_RESULT_H_
#define LIBRARIES_DATA_LOCAL_ALIGNMENT_RESULT_H_

#include "rectangle.h"
#include "alignment_result.h"
#include "core/optional.h"
#include "bspline_grid.h"

template<typename T>
struct FramePatchMeta {
    // rectangle representing the patch
    Rectangle<Point2D<T>> rec;
    // logical id of the patch
    size_t id_x;
    size_t id_y;
    size_t id_t;
};

template<typename T>
struct LocalAlignmentResult {
    const AlignmentResult<T> &globalHint;
    Dimensions movieDim;
    // these are shifts (including global shift) of all patches in X/Y dimension,
    // i.e. if you want to compensate for the shift,
    // you have to shift in opposite direction (negate these values)
    std::vector<std::pair<FramePatchMeta<T>, Point2D<T>>> shifts;
    core::optional<BSplineGrid<T>> bsplineRep;
};


#endif /* LIBRARIES_DATA_LOCAL_ALIGNMENT_RESULT_H_ */
