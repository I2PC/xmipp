
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

#include "sga_nn_online_pca.h"

#include <cassert>
#include <algorithm>

template<typename T>
SgaNnOnlinePca<T>::SgaNnOnlinePca(  size_t nComponents,
                                    size_t nPrincipalComponents,
                                    size_t initialBatchSize )
    : m_counter(0)
    , m_mean(nComponents)
    , m_centered(nComponents)
    , m_projection(nPrincipalComponents)
    , m_eigenValues(nPrincipalComponents)
    , m_eigenVectors(nComponents, nPrincipalComponents)
    , m_batch(nComponents, initialBatchSize)
    , m_eigenVectorUpdater(nComponents)
{
}

template<typename T>
size_t SgaNnOnlinePca<T>::getComponentCount() const {
    return MAT_YSIZE(m_eigenVectors);
}

template<typename T>
size_t SgaNnOnlinePca<T>::getPrincipalComponentCount() const {
    return MAT_XSIZE(m_eigenVectors);
}

template<typename T>
size_t SgaNnOnlinePca<T>::getInitialBatchSize() const {
    return MAT_XSIZE(m_batch);
}

template<typename T>
size_t SgaNnOnlinePca<T>::getSampleSize() const {
    return m_counter;
}

template<typename T>
void SgaNnOnlinePca<T>::getMean(Matrix1D<T>& v) const {
    v = m_mean;
}

template<typename T>
void SgaNnOnlinePca<T>::getAxisVariance(Matrix1D<T>& v) const {
    v = m_eigenValues;
}

template<typename T>
void SgaNnOnlinePca<T>::getBasis(Matrix2D<T>& b) const {
    b = m_eigenVectors;
}



template<typename T>
void SgaNnOnlinePca<T>::reset() {
    m_counter = 0;
}

template<typename T>
void SgaNnOnlinePca<T>::learn(const Matrix1D<T>& v, const T& gamma) {
    if (VEC_XSIZE(v) != getComponentCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Input vector has incorrect size");
    }

    if (m_counter < getInitialBatchSize()) {
        learnFirstFew(v);
    } else {
        learnOthers(v, gamma);
    }
}

template<typename T>
void SgaNnOnlinePca<T>::learn(const Matrix1D<T>& v) {
    learn(v, calculateGamma());
}

template<typename T>
void SgaNnOnlinePca<T>::finalize() {
    if(m_counter < getInitialBatchSize()) {
        // Initial batch has not been completed. 
        REPORT_ERROR(ERR_NUMERICAL, "Learn was not called sufficient times to fulfil intial batch");
    }
}



template<typename T>
void SgaNnOnlinePca<T>::center(Matrix1D<T>& v) const {
    if(!v.sameShape(m_mean)) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Input vector has incorrect size");
    }

    v -= m_mean;
}

template<typename T>
void SgaNnOnlinePca<T>::center(const Matrix1D<T>& v, Matrix1D<T>& c) const {
    if(!v.sameShape(m_mean)) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Input vector has incorrect size");
    }

    c.resizeNoCopy(v);
    FOR_ALL_ELEMENTS_IN_MATRIX1D(c) {
        VEC_ELEM(c, i) = VEC_ELEM(v, i) - VEC_ELEM(m_mean, i);
    }
}

template<typename T>
void SgaNnOnlinePca<T>::uncenter(Matrix1D<T>& v) const {
    if(!v.sameShape(m_mean)) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Input vector has incorrect size");
    }

    v += m_mean;
}

template<typename T>
void SgaNnOnlinePca<T>::uncenter(const Matrix1D<T>& c, Matrix1D<T>& v) const {
    v = c;
    uncenter(v); //TODO more efficient impl
}

template<typename T>
void SgaNnOnlinePca<T>::projectCentered(const Matrix1D<T>& v, Matrix1D<T>& p) const {
    if (VEC_XSIZE(v) != getComponentCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Input vector has incorrect size");
    }
    matrixOperation_Atx(m_eigenVectors, v, p);
}

template<typename T>
void SgaNnOnlinePca<T>::unprojectCentered(const Matrix1D<T>& p, Matrix1D<T>& v) const {
    if (VEC_XSIZE(p) != getPrincipalComponentCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Input vector has incorrect size");
    }
    matrixOperation_Ax(m_eigenVectors, p, v);
}


template<typename T>
void SgaNnOnlinePca<T>::centerAndProject(Matrix1D<T>& v, Matrix1D<T>& p) const {
    center(v);
    projectCentered(v, p);
}

template<typename T>
void SgaNnOnlinePca<T>::unprojectAndUncenter(const Matrix1D<T>& p, Matrix1D<T>& v) const {
    unprojectCentered(p, v);
    uncenter(v);
}





template<typename T>
T SgaNnOnlinePca<T>::calculateGamma() const {
    constexpr auto c = T(1); //TODO determine 
    return c / std::sqrt(m_counter);
}

template<typename T>
void SgaNnOnlinePca<T>::learnFirstFew(const Matrix1D<T>& v) {
    assert(m_counter < getInitialBatchSize());

    // Store the arriving vector
    m_batch.setCol(m_counter++, v);

    // Finalize when called for the last time
    if(m_counter == getInitialBatchSize()) {
        // Set the mean
        m_batch.computeRowMeans(m_mean);
        assert(VEC_XSIZE(m_mean) == MAT_YSIZE(m_batch));

        // Subtract the mean to each column
        subtractToAllColumns(m_batch, m_mean);

        // Compute a batch PCA with the vectors
        batchPca(m_batch, m_eigenVectors, m_eigenValues, getPrincipalComponentCount());
    }
}

template<typename T>
void SgaNnOnlinePca<T>::learnOthers(const Matrix1D<T>& v, const T& gamma) {
    updateMean(m_mean, m_counter, v);
    center(v, m_centered);
    projectCentered(m_centered, m_projection);
    updateEigenValues(m_eigenValues, m_projection, gamma);
    m_eigenVectorUpdater(m_eigenVectors, m_centered, m_projection, gamma);

    // Increment the counter
    ++m_counter;
}



template<typename T>
void SgaNnOnlinePca<T>::updateMean(Matrix1D<T>& mean, size_t count, const Matrix1D<T>& v) {
    // n/(n + 1) * old + 1/(n + 1) * v
    assert(mean.sameShape(v));

    mean *= count;
    mean += v;
    mean /= count + 1;
}

template<typename T>
void SgaNnOnlinePca<T>::updateEigenValues(  Matrix1D<T>& values, 
                                            const Matrix1D<T>& projection, 
                                            const T& gamma )
{
    //values' = values + gamma*(projection^2 - values)
    // = (1 - gamma)*values + gamma*projection^2
    assert(values.sameShape(projection));

    values *= (1 - gamma);

    FOR_ALL_ELEMENTS_IN_MATRIX1D(values) {
        const auto& g = VEC_ELEM(projection, i);
        auto& l = VEC_ELEM(values, i);
        l += gamma*g*g;
    }
}



template<typename T>
SgaNnOnlinePca<T>::EigenVectorUpdater::EigenVectorUpdater(size_t nRows)
    : m_column(nRows)
    , m_sigma(nRows)
    , m_aux(nRows)
{
}

template<typename T>
void SgaNnOnlinePca<T>::EigenVectorUpdater::operator()( Matrix2D<T>& vectors, 
                                                        const Matrix1D<T>& centered, 
                                                        const Matrix1D<T>& projection, 
                                                        const T& gamma )
{
    // vector_i' = vector_i + gamma*projection_i*(centered - projection_i*vector_i + 2*sum(projection_j*vector_j, j=1:j-1))
    // = vector_i + gamma*projection_i*(centered + projection_i*vector_i + 2*sum(projection_j*vector_j, j=1:i)

    // Shorthands
    const auto nCols =  MAT_XSIZE(vectors);
    const auto nRows =  MAT_YSIZE(vectors);
    assert(VEC_XSIZE(centered) == nRows);
    assert(VEC_XSIZE(projection) == nCols);

    // Aux variables
    auto& column = m_column; // Contains the current working column
    auto& sigma = m_sigma; // Contains 2*sum(projection_j*vector_j, j=1:i)
    auto& aux = m_aux; // Contains the term to be added to the column for updating it

    // Compute the basis column by column
    sigma.initZeros(); // Initialize the sum to zeros
    for (size_t i = 0; i < nCols; ++i) {
        // Get the vector and projection for this iteration
        vectors.getCol(i, column);
        const auto& phi = VEC_ELEM(projection, i);

        // Update sigma (add phi_i*column_i to it to consider the sum
        // of all elements upto i)
        aux = column;
        aux *= (2*phi); //aux = 2*phi*column
        sigma += aux; //sigma = 2*sum(projection_j*vector_j, j=1:i)

        // Calculate the new column value
        // We have 2*phi*column in aux as an starting point
        aux /= 2; //aux = phi*column
        aux += centered; // aux = centered + phi*column
        aux -= sigma; // aux = centered + phi*column - 2*sum(projection_j*vector_j, j=1:i)
        aux *= gamma*phi; // aux = gamma*phi*(centered + phi*column - 2*sum(...))
        column += aux;
        
        //Make the column length 1
        column.selfNormalize();

        // Update the column
        vectors.setCol(i, column);
    }
}

template<typename T>
void SgaNnOnlinePca<T>::batchPca(   const Matrix2D<T>& batch, 
                                    Matrix2D<T>& vectors, 
                                    Matrix1D<T>& values, 
                                    size_t nPrincipalComponents ) 
{
    // Determine if thee batch needs to be transposed for 
    // faster computations.
    const auto transpose = MAT_XSIZE(batch) < MAT_YSIZE(batch);

    // Compute the covariance matrix of the batch
    // assuming it has been mean centered.
    Matrix2D<T> covariance;
    if(transpose) {
        matrixOperation_AtA(batch, covariance);
    } else {
        matrixOperation_AAt(batch, covariance);
    }
    covariance /= MAT_XSIZE(batch) - 1;

    // Due to the transposition, we should have
    // a the minimal size for the covariance matrix
    assert(MAT_XSIZE(covariance) == std::min(MAT_XSIZE(batch), MAT_YSIZE(batch)));
    assert(MAT_YSIZE(covariance) == std::min(MAT_XSIZE(batch), MAT_YSIZE(batch)));

    // Compute the first N eigenvectors with the highest eigenvalues
    if(transpose) {
        Matrix2D<T> temp;
        firstEigs(covariance, nPrincipalComponents, values, temp);
        assert(MAT_XSIZE(batch) == MAT_YSIZE(temp));
        matrixOperation_AB(batch, temp, vectors);
        normalizeColumnLengths(vectors);
    } else {
        firstEigs(covariance, nPrincipalComponents, values, vectors);
    }

    // Ensure the eigenvalues are in descending order
    assert(std::is_sorted(
        MATRIX1D_ARRAY(values), 
        MATRIX1D_ARRAY(values) + VEC_XSIZE(values), 
        std::greater<double>()
    ));
}

// Explicit instantiation
template class SgaNnOnlinePca<double>;
//template class SgaNnOnlinePca<float>;