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

            for(int controlIdxT = -1; controlIdxT < (lT - 1); ++controlIdxT) {
                T tmpT = Bspline03((tileCenterT / hT) - controlIdxT);
                if (tmpT == (T)0) continue;
                for(int controlIdxY = -1; controlIdxY < (lY - 1); ++controlIdxY) {
                    T tmpY = Bspline03((tileCenterY / hY) - controlIdxY);
                    if (tmpY == (T)0) continue;
                    for(int controlIdxX = -1; controlIdxX < (lX - 1); ++controlIdxX) {
                        T tmpX = Bspline03((tileCenterX / hX) - controlIdxX);
                        T val = tmpT * tmpY * tmpX;
                        int j = ((controlIdxT + 1) * lX * lY) +
                                ((controlIdxY + 1) * lX) + (controlIdxX + 1);
                        MAT_ELEM(A,tileIdxT*noOfPatchesXY + i, j) = val;
                    }
                }
            }
            VEC_ELEM(bX,tileIdxT*noOfPatchesXY + i) = -shift.x; // we want the BSPline describing opposite transformation,
            VEC_ELEM(bY,tileIdxT*noOfPatchesXY + i) = -shift.y; // so that we can use it to compensate for the shift
        }

        // solve the equation system for the spline coefficients
        Matrix1D<T> coefsX;
        Matrix1D<T> coefsY;
        EquationSystemSolver::solve(bX, bY, A, coefsX, coefsY, verbosity + 1, solverIters);
        return std::make_pair(coefsX, coefsY);
}

template
std::pair<float, float> BSplineHelper::getShift(const BSplineGrid<float> &grid, Dimensions dim,
        size_t x, size_t y, size_t n);
template
std::pair<double, double> BSplineHelper::getShift(const BSplineGrid<double> &grid, Dimensions dim,
        size_t x, size_t y, size_t n);
template<typename T>
std::pair<T, T> BSplineHelper::getShift(const BSplineGrid<T> &grid, Dimensions dim,
        size_t x, size_t y, size_t n) {
    T shiftX = 0;
    T shiftY = 0;
    getShift(grid.getDim().x(), grid.getDim().y(), grid.getDim().n(),
            dim.x(), dim.y(), dim.n(),
            x, y, n,
            shiftX, shiftY,
            grid.getCoeffsX().vdata, grid.getCoeffsY().vdata);

    return std::make_pair(shiftX, shiftY);
}

template<typename T>
void BSplineHelper::getShift(int lX, int lY, int lN,
        int xdim, int ydim, int ndim,
        int x, int y, int n,
        T &shiftY, T &shiftX,
        const T *coeffsX, const T *coeffsY) {
    using std::max;
    using std::min;

    T delta = 0.0001;
    // take into account end points
    T hX = (lX == 3) ? xdim : (xdim / (T) ((lX - 3)));
    T hY = (lY == 3) ? ydim : (ydim / (T) ((lY - 3)));
    T hT = (lN == 3) ? ndim : (ndim / (T) ((lN - 3)));
    // index of the 'cell' where pixel is located (<0, N-3> for N control points)
    T xPos = x / hX;
    T yPos = y / hY;
    T tPos = n / hT;
    // indices of the control points are from -1 .. N-2 for N points
    // pixel in 'cell' 0 may be influenced by points with indices <-1,2>
    for (int idxT = max(-1, (int) (tPos) - 1);
            idxT <= min((int) (tPos) + 2, lN - 2);
            ++idxT) {
        T tmpT = Bspline03(tPos - idxT);
        for (int idxY = max(-1, (int) (yPos) - 1);
                idxY <= min((int) (yPos) + 2, lY - 2);
                ++idxY) {
            T tmpY = Bspline03(yPos - idxY);
            for (int idxX = max(-1, (int) (xPos) - 1);
                    idxX <= min((int) (xPos) + 2, lX - 2);
                    ++idxX) {
                T tmpX = Bspline03(xPos - idxX);
                T tmp = tmpX * tmpY * tmpT;
                if (fabsf(tmp) > delta) {
                    size_t coeffOffset = (idxT + 1) * (lX * lY)
                            + (idxY + 1) * lX + (idxX + 1);
                    shiftX += coeffsX[coeffOffset] * tmp;
                    shiftY += coeffsY[coeffOffset] * tmp;
                }
            }
        }
    }
}
