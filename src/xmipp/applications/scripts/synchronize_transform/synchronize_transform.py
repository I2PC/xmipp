#!/usr/bin/env python

# ***************************************************************************
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************/

from typing import List, Optional

import argparse
import numpy as np
import scipy.linalg
import scipy.sparse
import cvxpy as cp

def check_graph(graph: scipy.sparse.spmatrix,
                k: int ) -> int:
    if graph.shape[0] != graph.shape[1]:
        raise RuntimeError("Graph matrix must be square")
      
    n = graph.shape[0] // k
    if n*k != graph.shape[0]:
        raise RuntimeError("Graph must have a size multiple of k")
    
    return n

def check_weights(graph: scipy.sparse.spmatrix,
                  weights: scipy.sparse.spmatrix ):
    if graph.shape != weights.shape:
        raise RuntimeError("Weights must match graph size")

def get_optimization_objective_function(p: cp.Variable,
                                        graph: scipy.sparse.spmatrix,
                                        weights: Optional[scipy.sparse.spmatrix] = None,
                                        triangular_upper: bool = False ) -> cp.Expression:
    # Convert to coordinate format
    if triangular_upper:
        graph_coo = scipy.sparse.triu(graph, format='coo')
    else:
        graph_coo = graph.tocoo()

    # Obtain items
    positions = (graph_coo.row, graph_coo.col)
    values = graph_coo.data
    
    # Ensemble the cost function
    if weights is None:
        return cp.sum_squares(p[positions] - values)
    else:
        weights = weights.tocsr()
        weights_sqrt = np.sqrt(weights[positions].A1)
        return cp.sum_squares(cp.multiply(weights_sqrt, (p[positions] - values)))

def get_optimization_constraints(p: cp.Variable,
                                 n: int,
                                 k: int,
                                 norm: Optional[str] = None ) -> List[cp.Constraint]:
    constraints = [p >> 0] # Semi-definite positive
    
    if norm == '1':
        constraints.append(cp.diag(p) == 1)
        
    elif norm == 'O' or norm == 'SO':
        # Diagonal blocks with identity
        for i in range(n):
            start = k*i
            end = start+k
            constraints.append(p[start:end, start:end] == np.eye(k))
        
    return constraints

def optimize_pairwise_matrix(graph: scipy.sparse.spmatrix,
                             n: int,
                             k: int,
                             weights: Optional[scipy.sparse.spmatrix] = None,
                             verbose: bool = False,
                             norm: Optional[str] = None,
                             triangular_upper: bool = False ) -> np.ndarray:
    p = cp.Variable(graph.shape, symmetric=True)
    objective_function = get_optimization_objective_function(
        p=p,
        graph=graph,
        weights=weights,
        triangular_upper=triangular_upper
    )
    constraints = get_optimization_constraints(p=p, n=n, k=k, norm=norm)
    
    problem = cp.Problem(cp.Minimize(objective_function), constraints)
    objective_value = problem.solve(verbose=verbose)
    
    return p.value, objective_value

def orthogonalize(matrices: np.ndarray, special: bool = False) -> np.ndarray:
    u, s, vt = np.linalg.svd(matrices, full_matrices=False)
    if not special:
        u *= np.sign(s)[...,None,:]
    return u @ vt
    
def compute_bases(pairwise: np.ndarray,
                  n: int,
                  k: int,
                  norm: Optional[str] = None ) -> np.ndarray:
    w, v = scipy.linalg.eigh(pairwise, subset_by_index=[k*n-k, k*n-1])
    v *= np.sqrt(n)
    v = v.reshape(n, k, k)
    
    if norm == 'O':
        v = orthogonalize(v, special=False)
    elif norm == 'SO':
        v = orthogonalize(v, special=True)
    
    return v, w

def main(input_graph_path: str,
         output_bases: str,
         k: int,
         weight_path: Optional[str] = None, 
         verbose: bool = False,
         norm: Optional[str] = None,
         triangular_upper: bool = False ):
    
    graph = scipy.sparse.load_npz(input_graph_path)
    n = check_graph(graph=graph, k=k)

    weights = None
    if weight_path is not None:
        weights = scipy.sparse.load_npz(weight_path)
        check_weights(graph=graph, weights=weights)
    
    pairwise, error = optimize_pairwise_matrix(
        graph=graph, 
        n=n, 
        k=k, 
        weights=weights,
        verbose=verbose,
        norm=norm,
        triangular_upper=triangular_upper
    )
    print('Matrix completion error (ideally 0): ', error)
    
    bases, eigen_values = compute_bases(
        pairwise=pairwise,
        n=n,
        k=k,
        norm=norm
    )
    eigen_values /= n
    print('Matrix decomposition\'s normalized eigenvalues (ideally 1s): ', eigen_values)
    
    np.save(output_bases, bases)
    
if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(prog='Synchronize bases on a graph')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('-k', required=True, type=int)
    parser.add_argument('-w')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--norm')
    parser.add_argument('--triangular_upper', action='store_true')

    # Parse
    args = parser.parse_args()

    # Run the program
    main(
        input_graph_path=args.i,
        output_bases=args.o,
        k=args.k,
        weight_path=args.w,
        verbose=args.verbose,
        norm=args.norm,
        triangular_upper=args.triangular_upper
    )
    