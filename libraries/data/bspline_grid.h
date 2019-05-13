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

#ifndef LIBRARIES_DATA_BSPLINE_GRID_H_
#define LIBRARIES_DATA_BSPLINE_GRID_H_

#include "dimensions.h"
#include "core/matrix1d.h"

template<typename T>
class BSplineGrid {
public:
    BSplineGrid(Dimensions &dim, Matrix1D<T> &coeffsX, Matrix1D<T> &coeffsY):
        dim(dim), coeffsX(coeffsX), coeffsY(coeffsY) {}

    constexpr const Dimensions& getDim() const {
        return dim;
    }

    constexpr const Matrix1D<T>& getCoeffsX() const {
        return coeffsX;
    }

    constexpr const Matrix1D<T>& getCoeffsY() const {
        return coeffsY;
    }

private:
    Dimensions dim;
    Matrix1D<T> coeffsX;
    Matrix1D<T> coeffsY;
};

#endif /* LIBRARIES_DATA_BSPLINE_GRID_H_ */
