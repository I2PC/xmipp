/***************************************************************************
 *
 * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

#ifndef ONLINE_PCA_
#define ONLINE_PCA_

#include <core/matrix1d.h>
#include <core/matrix2d.h>

/**@defgroup OnlinePca Online PCA
   @ingroup DataLibrary */
//@{

/** Online PCA class using the Stochastic Gradien Ascend (SGA)
 * Neural Network (NN) approach
 *
 *  Example of use:
 *  @code
 *  SgaNnOnlinePca analyzer;
 *  FOR_ALL_VECTORS ...
 *     analyzer.addVector(v);
 *  @endcode
 *  */

template<typename T>
class SgaNnOnlinePca {
public:
   SgaNnOnlinePca(size_t nComponents, size_t nPrincipalComponents);
   SgaNnOnlinePca(const SgaNnOnlinePca& other) = default;
   ~SgaNnOnlinePca() = default;

   SgaNnOnlinePca& operator=(const SgaNnOnlinePca& other) = default;

   void reset();

   void learn(const Matrix1D<T>& v, const T& gamma);
   void learn(const Matrix1D<T>& v);

   void project(const Matrix1D<T>& v, Matrix1D<T>& p) const;

private:
   size_t m_counter;
   Matrix1D<T> m_mean;
   Matrix1D<T> m_gradient;
   Matrix1D<T> m_eigen;
   Matrix2D<T> m_basis;


   void learnFirst(const Matrix1D<T>& v);
   void learnOthers(const Matrix1D<T>& v, const T& gamma);

};

//@}
#endif