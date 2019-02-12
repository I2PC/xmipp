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

#ifndef LIBRARIES_RECONSTRUCTION_BSPLINE_HELPER_H_
#define LIBRARIES_RECONSTRUCTION_BSPLINE_HELPER_H_

#include "data/local_alignment_result.h"
#include "data/dimensions.h"
#include "eq_system_solver.h"
#include "core/matrix2d.h"
#include <iosfwd>
#include <utility> // std::make_pair
#include <type_traits>

class BSplineHelper {
public:
    /**
     * Computes BSpline coefficients from given data
     * @param movieSize
     * @param alignment to use
     * @param controlPoints of the resulting spline
     * @param noOfPatches used for generating the alignment
     * @param verbosity level
     * @param solverIters max iterations of the solver
     * @return coefficients of the BSpline representing the local shifts
     */
    template<typename T>
    static std::pair<Matrix1D<T>, Matrix1D<T>> computeBSplineCoeffs(const Dimensions &movieSize,
        const LocalAlignmentResult<T> &alignment,
        const Dimensions &controlPoints, const std::pair<size_t, size_t> &noOfPatches,
        int verbosity, int solverIters);

    template<typename T>
    static constexpr inline T Bspline03(T argument)    {
        static_assert(std::is_same<float, T>::value
                || std::is_same<double, T>::value, "T must be either float or double");
        if (std::is_same<float, T>::value) {
            argument = fabsf(argument);
        } else {
            argument = fabs(argument);
        }
        if (argument < (T)1) {
            return(argument * argument * (argument - (T)2) * (T)0.5 + (T)2 / (T)3);
        }
        else if (argument < (T)2) {
            argument -= (T)2;
            return(argument * argument * argument * ((T)-1 / (T)6));
        }
        else {
            return (T)0;
        }
    }
};

#endif /* LIBRARIES_RECONSTRUCTION_BSPLINE_HELPER_H_ */
