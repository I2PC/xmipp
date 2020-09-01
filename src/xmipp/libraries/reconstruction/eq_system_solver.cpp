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

#include "eq_system_solver.h"
#include "core/linear_system_helper.h"

template void EquationSystemSolver::solve(Matrix1D<float>& bXt,
        Matrix1D<float>& bYt, Matrix2D<float>& At, Matrix1D<float>& shiftXt,
        Matrix1D<float>& shiftYt, int verbosity, int iterations);
template void EquationSystemSolver::solve(Matrix1D<double>& bXt,
        Matrix1D<double>& bYt, Matrix2D<double>& At, Matrix1D<double>& shiftXt,
        Matrix1D<double>& shiftYt, int verbosity, int iterations);
template<typename T>
void EquationSystemSolver::solve(Matrix1D<T>& bXt,
        Matrix1D<T>& bYt, Matrix2D<T>& At, Matrix1D<T>& shiftXt,
        Matrix1D<T>& shiftYt, int verbosity, int iterations) {
    Matrix1D<double> ex;
    Matrix1D<double> ey;
    WeightedLeastSquaresHelper helper;
    Matrix2D<double> A;
    Matrix1D<double> bX;
    Matrix1D<double> bY;
    Matrix1D<double> shiftX;
    Matrix1D<double> shiftY;
    typeCast(At, helper.A);
    typeCast(bXt, bX);
    typeCast(bYt, bY);
    typeCast(shiftXt, shiftX);
    typeCast(shiftYt, shiftY);

    helper.w.initZeros(VEC_XSIZE(bX));
    helper.w.initConstant(1);

    int it = 0;
    double mean;
    double varbX;
    double varbY;
    bX.computeMeanAndStddev(mean, varbX);
    varbX *= varbX;
    bY.computeMeanAndStddev(mean, varbY);
    varbY *= varbY;
    if (verbosity > 1)
        std::cout << "Solving equation system ...\n";
    do {
        // Solve the equation system
        helper.b = bX;
        weightedLeastSquares(helper, shiftX);
        helper.b = bY;
        weightedLeastSquares(helper, shiftY);

        // Compute residuals
        ex = bX - helper.A * shiftX;
        ey = bY - helper.A * shiftY;

        // Compute R2
        double vareX;
        ex.computeMeanAndStddev(mean, vareX);
        vareX *= vareX;
        double vareY;
        ey.computeMeanAndStddev(mean, vareY);
        vareY *= vareY;
        double R2x = 1 - vareX / varbX;
        double R2y = 1 - vareY / varbY;
        if (verbosity > 1)
            std::cout << "Iteration " << it << " R2x=" << R2x << " R2y=" << R2y
                    << std::endl;

        // Identify outliers
        double oldWeightSum = helper.w.sum();
        double stddeveX = sqrt(vareX);
        double stddeveY = sqrt(vareY);
        FOR_ALL_ELEMENTS_IN_MATRIX1D (ex)
            if (fabs(VEC_ELEM(ex, i)) > 3 * stddeveX
                    || fabs(VEC_ELEM(ey, i)) > 3 * stddeveY)
                VEC_ELEM(helper.w, i) = 0.0;
        double newWeightSum = helper.w.sum();
        if ((newWeightSum == oldWeightSum) && (verbosity > 1)){
            std::cout << "No outlier found\n\n";
            break;
        } else if (verbosity > 1)
            std::cout << "Found " << (int) (oldWeightSum - newWeightSum)
                    << " outliers\n\n";

        it++;
    } while (it < iterations);

    typeCast(shiftX, shiftXt);
    typeCast(shiftY, shiftYt);
}
