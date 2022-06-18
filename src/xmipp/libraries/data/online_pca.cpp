
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

template<typename T>
SgaNnOnlinePca<T>::SgaNnOnlinePca(size_t nComponents, size_t nPrincipalComponents)
    : m_mean(nComponents)
    , m_gradient(nPrincipalComponents)
    , m_eigen(nPrincipalComponents)
    , m_basis(nComponents, nPrincipalComponents)
{
}

template<typename T>
void SgaNnOnlinePca<T>::reset() {
    m_counter = 0;
}

template<typename T>
void SgaNnOnlinePca<T>::learn(const Matrix1D<T>& v, const T& gamma) {
    if (m_counter > 0) {
        learnOthers(v, gamma);
    } else {
        learnFirst(v);
    }
}

template<typename T>
void SgaNnOnlinePca<T>::learn(const Matrix1D<T>& v) {
    constexpr T c = 1; //TODO determine
    learn(v, c/m_counter);
}

template<typename T>
void SgaNnOnlinePca<T>::project(const Matrix1D<T>& v, Matrix1D<T>& p) const {
    matrixOperation_Atx(m_basis, v, p);
}



template<typename T>
void SgaNnOnlinePca<T>::learnFirst(const Matrix1D<T>& v) {
    // The mean of one element is itself
    m_mean = v;

    // Gradient of the first iteration is 0, as v - mu_v is 0
    m_gradient.initZeros();

    // TODO idk which initial eigenvalues/eigenvectors to use at the beggining

    // Increment the counter
    m_counter = 1;
}

template<typename T>
void SgaNnOnlinePca<T>::learnOthers(const Matrix1D<T>& v, const T& gamma) {
    //TODO

    // Increment the counter
    ++m_counter;
}



// Explicit instantiation
template class SgaNnOnlinePca<double>;
//template class SgaNnOnlinePca<float>;