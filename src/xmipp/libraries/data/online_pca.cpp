
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

#include "online_pca.h"

#include <cassert>

template<typename T>
SgaNnOnlinePca<T>::SgaNnOnlinePca(size_t nComponents, size_t nPrincipalComponents)
    : m_mean(nComponents)
    , m_centered(nComponents)
    , m_gradient(nPrincipalComponents)
    , m_eigenValues(nPrincipalComponents)
    , m_eigenVectors(nComponents, nPrincipalComponents)
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
void SgaNnOnlinePca<T>::reset() {
    m_counter = 0;
}

template<typename T>
void SgaNnOnlinePca<T>::learn(const Matrix1D<T>& v, const T& gamma) {
    if (VEC_XSIZE(v) != getComponentCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Input vector has incorrect size");
    }

    if (m_counter > 0) {
        learnOthers(v, gamma);
    } else {
        learnFirst(v);
    }
}

template<typename T>
void SgaNnOnlinePca<T>::learn(const Matrix1D<T>& v) {
    learn(v, calculateGamma());
}

template<typename T>
void SgaNnOnlinePca<T>::learnNoEigenValues(const Matrix1D<T>& v, const T& gamma) {
    if (VEC_XSIZE(v) != getComponentCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Input vector has incorrect size");
    }

    if (m_counter > 0) {
        learnOthersNoEigenValues(v, gamma);
    } else {
        learnFirstNoEigenValues(v);
    }
}

template<typename T>
void SgaNnOnlinePca<T>::learnNoEigenValues(const Matrix1D<T>& v) {
    learnNoEigenValues(v, calculateGamma());
}

template<typename T>
void SgaNnOnlinePca<T>::project(const Matrix1D<T>& v, Matrix1D<T>& p) const {
    if (VEC_XSIZE(v) != getComponentCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Input vector has incorrect size");
    }
    matrixOperation_Atx(m_eigenVectors, v, p);
}

template<typename T>
void SgaNnOnlinePca<T>::unproject(const Matrix1D<T>& p, Matrix1D<T>& v) const {
    if (VEC_XSIZE(p) != getPrincipalComponentCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Input vector has incorrect size");
    }
    matrixOperation_Ax(m_eigenVectors, p, v);
}



template<typename T>
T SgaNnOnlinePca<T>::calculateGamma() const {
    constexpr auto c = T(1); //TODO determine 
    return c / m_counter;
}

template<typename T>
void SgaNnOnlinePca<T>::learnFirst(const Matrix1D<T>& v) {
    learnFirstNoEigenValues(v);
    m_eigenValues.initZeros(); //TODO IDK if it is ok
}

template<typename T>
void SgaNnOnlinePca<T>::learnFirstNoEigenValues(const Matrix1D<T>& v) {
    // The mean of one element is itself
    assert(m_mean.sameShape(v));
    m_mean = v;

    // Mean-centered vector of the first iteration is zero, as
    // the mean is the first vector itself
    m_centered.initZeros();

    // The gradient of the first iteration is also zero because
    // it depends on the dot product with the centered vector
    m_gradient.initZeros();

    // Initialize the eigenvectors (basis) with random values. 
    // TODO determine with COSS if OK
    m_eigenVectors.initGaussian(MAT_YSIZE(m_eigenVectors), MAT_XSIZE(m_eigenVectors), 0, 1);

    // Increment the counter
    m_counter = 1;
}

template<typename T>
void SgaNnOnlinePca<T>::learnOthers(const Matrix1D<T>& v, const T& gamma) {
    learnOthersNoEigenValues(v, gamma);
    updateEigenValues(m_eigenValues, m_gradient, gamma);
}

template<typename T>
void SgaNnOnlinePca<T>::learnOthersNoEigenValues(const Matrix1D<T>& v, const T& gamma) {
    updateMean(m_mean, m_counter, v);
    updateMeanCentered(m_centered, m_mean, v);
    updateGradient(m_gradient, m_eigenVectors, m_centered);
    m_eigenVectorUpdater(m_eigenVectors, m_centered, m_gradient, gamma);

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
void SgaNnOnlinePca<T>::updateMeanCentered( Matrix1D<T>& centered, 
                                            const Matrix1D<T>& mean, 
                                            const Matrix1D<T>& v) 
{
    assert(centered.sameShape(mean));
    assert(centered.sameShape(v));

    // centered = v - mean
    FOR_ALL_ELEMENTS_IN_MATRIX1D(centered) {
        VEC_ELEM(centered, i) = VEC_ELEM(v, i) - VEC_ELEM(mean, i);
    }
}

template<typename T>
void SgaNnOnlinePca<T>::updateGradient( Matrix1D<T>& gradient, 
                                        const Matrix2D<T>& basis, 
                                        const Matrix1D<T>& centered ) 
{
    // Dot product of each columnn of the basis with the centered vector
    matrixOperation_Atx(basis, centered, gradient);
}

template<typename T>
void SgaNnOnlinePca<T>::updateEigenValues(  Matrix1D<T>& values, 
                                            const Matrix1D<T>& gradient, 
                                            const T& gamma )
{
    //values' = values + gamma*(gradient^2 - values)
    // = (1 - gamma)*values + gamma*gradient^2
    assert(values.sameShape(gradient));

    values *= (1 - gamma);

    FOR_ALL_ELEMENTS_IN_MATRIX1D(values) {
        const auto& g = VEC_ELEM(gradient, i);
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
                                                        const Matrix1D<T>& gradient, 
                                                        const T& gamma )
{
    // vector_i' = vector_i + gamma*gradient_i*(centered - gradient_i*vector_i + 2*sum(gradient_j*vector_j, j=1:j-1))
    // = vector_i + gamma*gradient_i*(centered + gradient_i*vector_i + 2*sum(gradient_j*vector_j, j=1:i)

    // Shorthands
    const auto nCols =  MAT_XSIZE(vectors);
    const auto nRows =  MAT_YSIZE(vectors);
    assert(VEC_XSIZE(centered) == nRows);
    assert(VEC_XSIZE(gradient) == nCols);

    // Aux variables
    auto& column = m_column; // Contains the current working column
    auto& sigma = m_sigma; // Contains 2*sum(gradient_j*vector_j, j=1:i)
    auto& aux = m_aux; // Contains the term to be added to the column for updating it

    // Compute the basis column by column
    sigma.initZeros(); // Initialize the sum to zeros
    for (size_t i = 0; i < nCols; ++i) {
        // Get the vector and gradient for this iteration
        vectors.getCol(i, column);
        const auto& phi = VEC_ELEM(gradient, i);

        // Update sigma (add phi_i*column_i to it to consider the sum
        // of all elements upto i)
        aux = column;
        aux *= (2*phi); //aux = 2*phi*column
        sigma += aux; //sigma = 2*sum(gradient_j*vector_j, j=1:i)

        // Calculate the new column value
        // We have 2*phi*column in aux as an starting point
        aux /= 2; //aux = phi*column
        aux += centered; // aux = centered + phi*column
        aux -= sigma; // aux = centered + phi*column - 2*sum(gradient_j*vector_j, j=1:i)
        aux *= gamma*phi; // aux = gamma*phi*(centered + phi*column - 2*sum(...))
        column += aux;
        
        // Update the column
        vectors.setCol(i, column);
    }
}

// Explicit instantiation
template class SgaNnOnlinePca<double>;
//template class SgaNnOnlinePca<float>;