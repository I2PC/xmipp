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

#ifndef LIBRARIES_RECONSTRUCTION_EQ_SYSTEM_SOLVER_H_
#define LIBRARIES_RECONSTRUCTION_EQ_SYSTEM_SOLVER_H_

template<typename T>
class Matrix1D;
template<typename T>
class Matrix2D;

class EquationSystemSolver {
public:

    /**
     * Method computes absolute shifts from relative shifts
     * @param bX relative shifts in X dim
     * @param bY relative shifts in Y dim
     * @param A system matrix to be used
     * @param shiftX absolute shifts in X dim
     * @param shiftY absolute shifts in Y dim
     * @param verbosity level
     * @param iterations of the solver
     */
    template<typename T>
    static void solve(Matrix1D<T>& bXt,
            Matrix1D<T>& bYt, Matrix2D<T>& At, Matrix1D<T>& shiftXt,
            Matrix1D<T>& shiftYt, int verbosity, int iterations);
};

#endif /* LIBRARIES_RECONSTRUCTION_EQ_SYSTEM_SOLVER_H_ */
