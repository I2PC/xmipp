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


#ifndef LIBRARIES_DATA_ALIGNMENT_RESULT_H_
#define LIBRARIES_DATA_ALIGNMENT_RESULT_H_

#include "point2D.h"
#include <vector>
#include <cstddef>

/**@defgroup AlignmentResult Alignment Result
   @ingroup DataLibrary */
//@{
template<typename T>
struct AlignmentResult {
    size_t refFrame;
    // these are shifts from the reference frame in X/Y dimension,
    // i.e. if you want to compensate for the shift,
    // you have to shift in opposite direction (negate these values)
    std::vector<Point2D<T>> shifts;
};
//@}



#endif /* LIBRARIES_DATA_ALIGNMENT_RESULT_H_ */
