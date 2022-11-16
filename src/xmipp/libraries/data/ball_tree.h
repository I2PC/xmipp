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

#ifndef _CORE_BALL_TREE_H
#define _CORE_BALL_TREE_H

#include "core/matrix1d.h"
#include "core/matrix2d.h"

#include <vector>
#include <functional>

template<typename T>
class BallTree {
public:
    using Real = T;

    BallTree(const Matrix2D<Real>& samples);
    BallTree(const BallTree& other) = default;
    ~BallTree() = default;

    BallTree& operator=(const BallTree& other) = default;

    size_t nearest(const Matrix1D<Real>& point, Real& distance) const;
    //void query(const Matrix2D<Real>& points, std::vector<size_t>& ind);

private:
    class Node {
    public:
        Node(   const Real* data = nullptr, 
                const Node* left = nullptr, 
                const Node* right = nullptr,
                size_t cutAxis = 0, 
                Real radius = 0 );
        Node(const Node& other) = default;
        ~Node() = default;

        Node& operator=(const Node& other) = default;
    
        void setPoint(const Real* data);
        const Real* getPoint() const;

        void setLeft(const Node* node);
        const Node* getLeft() const;

        void setRight(const Node* node);
        const Node* getRight() const;

        void setCutAxis(size_t cut);
        size_t getCutAxis() const;

        void setRadius(Real radius);
        Real getRadius() const;

    private:
        const Real* m_point;
        const Node* m_left;
        const Node* m_right;
        size_t m_cutAxis;
        Real m_radius;
    };

    class NodeAxisCmp;

    using NodeVector = std::vector<Node>;

    std::reference_wrapper<const Matrix2D<Real>> m_samples;
    NodeVector m_nodes;
    const Node* m_root;

    static void fillNodes(const Matrix2D<Real>& points, NodeVector& v);
    static Node* buildTree( typename NodeVector::iterator begin, 
                            typename NodeVector::iterator end, 
                            size_t dim );
    static void nearestSearch(  const Node* root, 
                                const Real* point, 
                                size_t dim,
                                std::pair<const Node*, Real>& best );
    static void nearestSearch(  const Node* root, 
                                const Real* point, 
                                Real dist,
                                size_t dim,
                                std::pair<const Node*, Real>& best );
    static size_t calculateNodeCutAxis( typename NodeVector::const_iterator begin, 
                                        typename NodeVector::const_iterator end,
                                        size_t dim );
    static Real calculateNodeRadius(    typename NodeVector::const_iterator begin, 
                                        typename NodeVector::const_iterator middle,
                                        typename NodeVector::const_iterator end,
                                        size_t dim );

};

#endif //_CORE_BALL_TREE_H