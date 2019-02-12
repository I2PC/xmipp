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

#include "bspline_helper.h"

template
std::pair<Matrix1D<float>, Matrix1D<float>> BSplineHelper::computeBSplineCoeffs(const Dimensions &movieSize,
        const LocalAlignmentResult<float> &alignment,
        const Dimensions &controlPoints, const std::pair<size_t, size_t> &noOfPatches,
        int verbosity, int solverIters);
template<typename T>
std::pair<Matrix1D<T>, Matrix1D<T>> BSplineHelper::computeBSplineCoeffs(const Dimensions &movieSize,
        const LocalAlignmentResult<T> &alignment,
        const Dimensions &controlPoints, const std::pair<size_t, size_t> &noOfPatches,
        int verbosity, int solverIters) {
    if(verbosity) std::cout << "Computing BSpline coefficients" << std::endl;
        // get coefficients of the BSpline that can represent the shifts (formula  from the paper)
        int lX = controlPoints.x();
        int lY = controlPoints.y();
        int lT = controlPoints.n();
        int noOfPatchesXY = noOfPatches.first * noOfPatches.second;
        Matrix2D<T>A(noOfPatchesXY*movieSize.n(), lX * lY * lT);
        Matrix1D<T>bX(noOfPatchesXY*movieSize.n());
        Matrix1D<T>bY(noOfPatchesXY*movieSize.n());
        T hX = (lX == 3) ? movieSize.x() : (movieSize.x() / (T)(lX-3));
        T hY = (lY == 3) ? movieSize.y() : (movieSize.y() / (T)(lY-3));
        T hT = (lT == 3) ? movieSize.n() : (movieSize.n() / (T)(lT-3));

        for (auto &&r : alignment.shifts) {
            auto meta = r.first;
            auto shift = r.second;
            int tileIdxT = meta.id_t;
            int tileCenterT = tileIdxT * 1 + 0 + 0;
            int tileIdxX = meta.id_x;
            int tileIdxY = meta.id_y;
            int tileCenterX = meta.rec.getCenter().x;
            int tileCenterY = meta.rec.getCenter().y;
            int i = (tileIdxY * noOfPatches.first) + tileIdxX;

            for (int j = 0; j < (lT * lY * lX); ++j) {
                int controlIdxT = (j / (lY * lX)) - 1;
                int XY = j % (lY * lX);
                int controlIdxY = (XY / lX) -1;
                int controlIdxX = (XY % lX) -1;
                // note: if control point is not in the tile vicinity, val == 0 and can be skipped
                T val = Bspline03((tileCenterX / hX) - controlIdxX) *
                        Bspline03((tileCenterY / hY) - controlIdxY) *
                        Bspline03((tileCenterT / hT) - controlIdxT);
                MAT_ELEM(A,tileIdxT*noOfPatchesXY + i,j) = val;
            }
            VEC_ELEM(bX,tileIdxT*noOfPatchesXY + i) = -shift.x; // we want the BSPline describing opposite transformation,
            VEC_ELEM(bY,tileIdxT*noOfPatchesXY + i) = -shift.y; // so that we can use it to compensate for the shift
        }

        // solve the equation system for the spline coefficients
        Matrix1D<T> coefsX, coefsY;
        EquationSystemSolver::solve(bX, bY, A, coefsX, coefsY, verbosity + 1, solverIters);
        return std::make_pair(coefsX, coefsY);
}
