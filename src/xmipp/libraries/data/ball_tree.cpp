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

#include "ball_tree.h"

#include <algorithm>
#include <numeric>
#include <cassert>

template<typename T>
static T dot(const T* x, const T* y, size_t n) {
    T result = 0;

    for(size_t i = 0; i < n; ++i) {
        result += x[i] * y[i];
    }

    return result;
}

template<typename T>
static T distance2(const T* x, const T* y, const T& xlen, const T& ylen, size_t n) {
    return xlen + ylen - 2*dot(x, y, n);
}

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
static T length2(const T* x, size_t n) {
    T result = 0;

    for(size_t i = 0; i < n; ++i) {
        result += x[i] * x[i];
    }

    return result;
}

template<typename T>
static T distance(const T* x, const T* y, const T& xlen, const T& ylen, size_t n) {
    return std::sqrt(distance2(x, y, xlen, ylen, n));
}

template<typename T>
static T distance(const T* x, const T* y, size_t n) {
    return std::sqrt(distance2(x, y, n));
}

template<typename T>
static T length(const T* x, size_t n) {
    return std::sqrt(length2(x, n));
}





template<typename T>
BallTree<T>::Node::Node(const Real* data, 
                        const Node* left, 
                        const Node* right,
                        size_t cutAxis, 
                        Real radius )
    : m_point(data)
    , m_left(left)
    , m_right(right)
    , m_cutAxis(cutAxis)
    , m_radius(radius)
{
}

template<typename T>
void BallTree<T>::Node::setPoint(const Real* data) {
    m_point = data;
}

template<typename T>
const typename BallTree<T>::Real* BallTree<T>::Node::getPoint() const {
    return m_point;
}

template<typename T>
void BallTree<T>::Node::setLeft(const Node* node) {
    m_left = node;
}

template<typename T>
const typename BallTree<T>::Node* BallTree<T>::Node::getLeft() const {
    return m_left;
}

template<typename T>
void BallTree<T>::Node::setRight(const Node* node) {
    m_right = node;
}

template<typename T>
const typename BallTree<T>::Node* BallTree<T>::Node::getRight() const {
    return m_right;
}

template<typename T>
void BallTree<T>::Node::setCutAxis(size_t cut) {
    m_cutAxis = cut;
}

template<typename T>
size_t BallTree<T>::Node::getCutAxis() const {
    return m_cutAxis;
}

template<typename T>
void BallTree<T>::Node::setRadius(Real radius) {
    m_radius = radius;
}

template<typename T>
typename BallTree<T>::Real BallTree<T>::Node::getRadius() const {
    return m_radius;
}





template<typename T>
class BallTree<T>::NodeAxisCmp {
public:
    NodeAxisCmp(size_t axis) : m_axis(axis) {}
    NodeAxisCmp(const NodeAxisCmp& other) = default;
    ~NodeAxisCmp() = default;

    NodeAxisCmp& operator=(const NodeAxisCmp& other) = default;

    bool operator()(const Node& x, const Node& y) const {
        return x.getPoint()[m_axis] < y.getPoint()[m_axis];
    }

private:   
    size_t m_axis;

};





template<typename T>
BallTree<T>::BallTree(const Matrix2D<Real>& samples)
    : m_samples(samples)
    , m_nodes()
    , m_root(nullptr)
{
    // Build the tree
    fillNodes(m_samples, m_nodes);
    m_root = buildTree(
        m_nodes.begin(), m_nodes.end(), 
        MAT_XSIZE(m_samples.get())
    );
}



template<typename T>
size_t BallTree<T>::nearest(const Matrix1D<Real>& point, Real& distance) const {
    const auto& samples = m_samples.get();
    if(VEC_XSIZE(point) != MAT_XSIZE(samples)) {
        // TODO exception
    }

    // Search
    std::pair<const Node*, Real> best(nullptr, distance);
    nearestSearch(m_root, MATRIX1D_ARRAY(point), MAT_XSIZE(samples), best);

    // Calculate the output
    size_t result = MAT_YSIZE(samples);
    if(best.first) {
        // Do pointer tricks to obtain the index
        const Real* const begin = MATRIX2D_ARRAY(samples);
        const Real* const match = best.first->getPoint();
        const auto count = std::distance(begin, match);
        assert(count >= 0);
        assert((count % MAT_XSIZE(samples)) == 0);
        const size_t index = static_cast<size_t>(count) / MAT_XSIZE(samples);

        // Elaborate the output
        distance = best.second;
        result = index;
    }

    return result;
}



template<typename T>
void BallTree<T>::fillNodes(const Matrix2D<Real>& points, NodeVector& v) {
    v.clear();
    v.reserve(MAT_YSIZE(points));

    for(size_t i = 0; i < MAT_YSIZE(points); ++i) {
        v.emplace_back(&MAT_ELEM(points, i, 0));
    }

    assert(v.size() == MAT_YSIZE(points));
}

template<typename T>
typename BallTree<T>::Node* BallTree<T>::buildTree( typename NodeVector::iterator begin, 
                                                    typename NodeVector::iterator end, 
                                                    size_t dim )
{
    Node* result = nullptr;

    const auto count = std::distance(begin, end);
    if(count > 1) {
        // Find the maximum variance axis
        const auto axis = calculateNodeCutAxis(begin, end, dim);

        // Get the middle point
        const auto middle = begin + (count/2);

        // Partially sort the range. Middle point will contain the median
        std::nth_element(
            begin, middle, end,
            NodeAxisCmp(axis)
        );

        // Build the output recursively
        middle->setLeft (buildTree(begin,    middle, dim));
        middle->setRight(buildTree(middle+1, end,    dim));
        middle->setCutAxis(axis);
        middle->setRadius(calculateNodeRadius(begin, middle, end, dim));
        result = &(*middle);
    } else if (count == 1) {
        result = &(*begin);
    }

    return result;
}

template<typename T>
void BallTree<T>::nearestSearch(const Node* root, 
                                const Real* point, 
                                size_t dim,
                                std::pair<const Node*, Real>& best )
{
    if(root) {
        const auto dist = distance(root->getPoint(), point, dim);
        nearestSearch(*root, point, dist, dim, best);
    }
}

template<typename T>
void BallTree<T>::nearestSearch(const Node& root, 
                                const Real* point, 
                                Real dist,
                                size_t dim,
                                std::pair<const Node*, Real>& best )
{
    // Check if it is a candidate
    if((dist - root.getRadius()) < best.second) {
        // Evaluate if it is the best candidate so far
        if(dist < best.second) {
            best = std::make_pair(&root, dist);
        }

        const auto* left = root.getLeft();
        const auto* right = root.getRight();
        if(left && right) {
            const auto leftDist  = distance(left->getPoint(),  point, dim);
            const auto rightDist = distance(right->getPoint(), point, dim);

            // Evaluate first the closest branch so that the furthest one
            // is likely to be pruned
            if(leftDist < rightDist) {
                nearestSearch(*left,  point, leftDist,  dim, best);
                nearestSearch(*right, point, rightDist, dim, best);
            } else {
                nearestSearch(*right, point, rightDist, dim, best);
                nearestSearch(*left,  point, leftDist,  dim, best);
            }
            
        } else {
            // Evaluate only a branch if there is any
            nearestSearch(left,  point, dim, best);
            nearestSearch(right, point, dim, best);
        }
    }
}

template<typename T>
size_t BallTree<T>::calculateNodeCutAxis(   typename NodeVector::const_iterator begin, 
                                            typename NodeVector::const_iterator end,
                                            size_t dim )
{
    std::pair<size_t, Real> best(0, 0);
    const auto count = std::distance(begin, end);

    // Select the axis with the largest variance
    for(size_t i = 0; i < dim; ++i) {
        // Obtain the mean
        const auto mean = std::accumulate(
            begin, end, Real(0),
            [i] (Real sum, const Node& node) -> Real {
                return sum + node.getPoint()[i];
            }
        ) / count;
        
        const auto var = std::accumulate(
            begin, end, Real(0),
            [i, mean] (Real sum, const Node& node) -> Real {
                const auto delta = node.getPoint()[i] - mean;
                const auto delta2 = delta*delta;
                return sum + delta2;
            }
        );
        
        // Update the result if necessary
        if(var > best.second) {
            best = std::make_pair(i, var);
        }
    }

    return best.first;
}

template<typename T>
typename BallTree<T>::Real BallTree<T>::calculateNodeRadius(typename NodeVector::const_iterator begin, 
                                                            typename NodeVector::const_iterator middle,
                                                            typename NodeVector::const_iterator end,
                                                            size_t dim )
{
    Real r = 0;

    // Compare with all the points after/before middle
    for(auto ite = begin; ite != middle; ++ite) {
        r = std::max(r, distance(middle->getPoint(), ite->getPoint(), dim));
    }
    for(auto ite = middle; ite != end; ++ite) {
        r = std::max(r, distance(middle->getPoint(), ite->getPoint(), dim));
    }

    return r;
}

// Explicit instantiation
template class BallTree<float>;
template class BallTree<double>;