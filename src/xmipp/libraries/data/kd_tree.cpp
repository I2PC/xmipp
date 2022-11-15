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

#include "kd_tree.h"

#include <algorithm>
#include <limits>
#include <cassert>

template<typename T>
static T distance2(const T* x, const T* y, size_t n) {
    T result = 0;

    for(size_t i = 0; i < n; ++i) {
        const auto delta = x[i] - y[i];
        const auto delta2 = delta * delta;
        result += delta2;
    }

    return result;
}


template<typename T>
KDTree<T>::Node::Node(  const Real* data, 
                        const Node* left, 
                        const Node* right )
    : m_point(data)
    , m_left(left)
    , m_right(right)
{
}

template<typename T>
void KDTree<T>::Node::setPoint(const Real* data) {
    m_point = data;
}

template<typename T>
const typename KDTree<T>::Real* KDTree<T>::Node::getPoint() const {
    return m_point;
}

template<typename T>
void KDTree<T>::Node::setLeft(const Node* node) {
    m_left = node;
}

template<typename T>
const typename KDTree<T>::Node* KDTree<T>::Node::getLeft() const {
    return m_left;
}

template<typename T>
void KDTree<T>::Node::setRight(const Node* node) {
    m_right = node;
}

template<typename T>
const typename KDTree<T>::Node* KDTree<T>::Node::getRight() const {
    return m_right;
}







template<typename T>
KDTree<T>::KDTree(const Matrix2D<Real>& samples)
    : m_samples(samples)
    , m_nodes()
    , m_root(nullptr)
{
    // Build the tree
    fillNodes(m_samples, m_nodes);
    m_root = buildTree(
        m_nodes.begin(), m_nodes.end(), 
        0, MAT_XSIZE(m_samples.get())
    );
}



template<typename T>
size_t KDTree<T>::nearest(const Matrix1D<Real>& point, Real& distance) const {
    const auto& samples = m_samples.get();
    if(VEC_XSIZE(point) != MAT_XSIZE(samples)) {
        // TODO exception
    }

    // Search
    std::pair<const Node*, Real> best(nullptr, std::numeric_limits<Real>::infinity());
    nearestSearch(m_root, MATRIX1D_ARRAY(point), 0, MAT_XSIZE(samples), best);

    // Do pointer tricks to obtain the index
    const Real* const begin = MATRIX2D_ARRAY(samples);
    const Real* const match = best.first->getPoint();
    const auto count = std::distance(begin, match);
    assert(count >= 0);
    assert((count % MAT_XSIZE(samples)) == 0);
    const size_t index = static_cast<size_t>(count) / MAT_XSIZE(samples);

    // Elaborate the output
    distance = best.second;
    return index;
}



template<typename T>
void KDTree<T>::fillNodes(const Matrix2D<Real>& points, NodeVector& v) {
    v.clear();
    v.reserve(MAT_YSIZE(points));

    for(size_t i = 0; i < MAT_YSIZE(points); ++i) {
        v.emplace_back(&MAT_ELEM(points, i, 0));
    }

    assert(v.size() == MAT_YSIZE(points));
}

template<typename T>
typename KDTree<T>::Node* KDTree<T>::buildTree( typename NodeVector::iterator begin, 
                                                typename NodeVector::iterator end, 
                                                size_t axis,
                                                size_t dim )
{
    Node* result = nullptr;

    const auto count = std::distance(begin, end);
    if(count > 0) {
        // Get the middle point
        const auto middle = begin + (count/2);

        // Partially sort the range
        std::nth_element(
            begin, middle, end,
            [axis] (const Node& x, const Node& y) -> bool {
                return x.getPoint()[axis] < y.getPoint()[axis];
            }
        );

        // Recursively call
        const auto nextAxis = (axis + 1) % dim;
        middle->setLeft (buildTree(begin,    middle, nextAxis, dim));
        middle->setRight(buildTree(middle+1, end,    nextAxis, dim));
        result = &(*middle);
    }

    return result;
}

template<typename T>
void KDTree<T>::nearestSearch(  const Node* root, 
                                const Real* point, 
                                size_t axis,
                                size_t dim,
                                std::pair<const Node*, Real>& best )
{
    if(root) {
        // Calculate the distance to the root
        const auto d = distance2(root->getPoint(), point, dim); 
        
        // Evaluate if it is a better candidate
        if(d < best.second) {
            best = std::make_pair(root, d);
        }

        // Evaluate which branch to choose
        const auto delta = root->getPoint()[axis] - point[axis];
        const auto delta2 = delta * delta;
        const auto nextAxis = (axis + 1) % dim;

        // Recursively call
        nearestSearch(delta > 0 ? root->getLeft() : root->getRight(), point, nextAxis, dim, best);
        if(delta2 < best.second) {
            nearestSearch(delta > 0 ? root->getRight() : root->getLeft(), point, nextAxis, dim, best);
        }
    }
}

// Explicit instantiation
template class KDTree<float>;
template class KDTree<double>;