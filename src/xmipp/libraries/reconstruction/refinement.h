/***************************************************************************
 *
 * Authors:     Roberto Marabini
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

#ifndef _REFINEMENT_HH
#define _REFINEMENT_HH

class Projection;
class CorrelationAux;
template<typename T>
class MultidimArray;

/**@defgroup ShiftRefinement refinement (Shift refinement)
   @ingroup ReconsLibrary */
//@{

/**Correlates two projections and finds the maximun of the correlation matrix.
*/
void calculate_and_find_correlation_max_proj(Projection const &proj1,
        Projection const &proj2,
        Projection & proj_tmp,
        double &shift_X, double &shift_Y,
        double const max_step,
        int ref_trans_after, int act_proj,
        CorrelationAux &aux);

/**Correlates two matrices  and finds the maximun of the correlation matrix.
   This center may not be at  an integer position.
   The routine works as follows:
   - Search for the maximun with pixel acuraccy inside the selfWindow
   - Calculate the gravity centre of the corelation
         in a neighborhood such as maximum/sqrt(2) > value
   - Look for the gravity centre in this neighborhood
   Note: The neighborhood is circular
*/
template <class T>
void calculate_and_find_correlation_max_mat(MultidimArray<T> const &mat1,
        MultidimArray<T> const &mat2,
        MultidimArray<T> & mat_temp,
        double &shift_X, double &shift_Y,
        double const max_step,
        CorrelationAux &aux);
//@}

#endif
