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

#ifndef SGA_NN_ONLINE_PCA_
#define SGA_NN_ONLINE_PCA_

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
 *     analyzer.learn(v);
 *  @endcode
 *  */

template<typename T>
class SgaNnOnlinePca {
public:
   SgaNnOnlinePca(size_t nComponents, size_t nPrincipalComponents, size_t initialBatchSize);
   SgaNnOnlinePca(const SgaNnOnlinePca& other) = default;
   ~SgaNnOnlinePca() = default;

   SgaNnOnlinePca& operator=(const SgaNnOnlinePca& other) = default;

   size_t getComponentCount() const;
   size_t getPrincipalComponentCount() const;
   size_t getInitialBatchSize() const;
   size_t getSampleSize() const;

   void getMean(Matrix1D<T>& v) const;
   void getAxisVariance(Matrix1D<T>& v) const;
   void getBasis(Matrix2D<T>& b) const;

   void reset();
   void learn(const Matrix1D<T>& v, const T& gamma);
   void learn(const Matrix1D<T>& v);
   void finalize();

   void center(Matrix1D<T>& v) const;
   void center(const Matrix1D<T>& v, Matrix1D<T>& c) const;
   void uncenter(Matrix1D<T>& v) const;
   void uncenter(const Matrix1D<T>& c, Matrix1D<T>& v) const;

   void projectCentered(const Matrix1D<T>& v, Matrix1D<T>& p) const;
   void unprojectCentered(const Matrix1D<T>& p, Matrix1D<T>& v) const;

   void centerAndProject(Matrix1D<T>& v, Matrix1D<T>& p) const;
   void unprojectAndUncenter(const Matrix1D<T>& p, Matrix1D<T>& v) const;

private:
   class EigenVectorUpdater {
   public:
      EigenVectorUpdater() = default;
      explicit EigenVectorUpdater(size_t nRows);
      EigenVectorUpdater(const EigenVectorUpdater& other) = default;
      ~EigenVectorUpdater() = default;
      
      EigenVectorUpdater& operator=(const EigenVectorUpdater& other) = default;

      void operator()(  Matrix2D<T>& vectors, 
                        const Matrix1D<T>& centered, 
                        const Matrix1D<T>& projection, 
                        const T& gamma );

   private:
      Matrix1D<T> m_column;
      Matrix1D<T> m_sigma;
      Matrix1D<T> m_aux;

   };

   size_t m_counter;
   Matrix1D<T> m_mean;
   Matrix1D<T> m_centered;
   Matrix1D<T> m_projection;
   Matrix1D<T> m_eigenValues;
   Matrix2D<T> m_eigenVectors;
   Matrix2D<T> m_batch;

   EigenVectorUpdater m_eigenVectorUpdater;

   T calculateGamma() const;
   void learnFirstFew(const Matrix1D<T>& v);
   void learnOthers(const Matrix1D<T>& v, const T& gamma);
   void learnOthersNoEigenValues(const Matrix1D<T>& v, const T& gamma);

   static void updateMean(Matrix1D<T>& mean, size_t count, const Matrix1D<T>& v);
   static void updateEigenValues(Matrix1D<T>& values, const Matrix1D<T>& projection, const T& gamma);
   static void batchPca(const Matrix2D<T>& batch, Matrix2D<T>& vectors, Matrix1D<T>& values, size_t nPrincipalComponents);

};

//@}
#endif