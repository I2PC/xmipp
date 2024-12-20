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

from typing import List, Optional, Tuple

import argparse
import bisect
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.stats
import cvxpy as cp

def check_graph(graph: scipy.sparse.spmatrix,
                k: int ) -> int:
    if graph.shape[0] != graph.shape[1]:
        raise RuntimeError("Graph matrix must be square")
      
    n = graph.shape[0] // k
    if n*k != graph.shape[0]:
        raise RuntimeError("Graph must have a size multiple of k")
    
    return n

def orthogonalize_matrices(matrices: np.ndarray, special: bool = False) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrices, full_matrices=False)
    result = u @ vh
    
    if special:
        sign = np.linalg.det(result)
        result[...,:,-1] *= sign
    
    return result

def ortho_group_synchronization_objective(samples: scipy.sparse.csr_matrix,
                                          adjacency: scipy.sparse.coo_matrix,
                                          transforms: np.ndarray,
                                          k: int) -> float:
    result = 0.0

    for i, j in zip(adjacency.row, adjacency.col):
        start0 = i*k
        end0 = start0+k
        start1 = j*k
        end1 = start1+k
        sample = samples[start0:end0, start1:end1].todense()
        delta = transforms[i] @ transforms[j].T
        result += np.trace(sample @ delta.T)

    return result

def calculate_pairwise_matrix(transforms: np.ndarray, 
                              n: int,
                              k: int,
                              k2: int ) -> np.ndarray:
    p = transforms.reshape(n*k, k2)
    return p @ p.T 

def get_sdp_objective(x: cp.Variable,
                      samples: scipy.sparse.csr_matrix,
                      adjacency: scipy.sparse.coo_matrix,
                      k: int ) -> cp.Expression:
    result = None
    
    for i, j in zip(adjacency.row, adjacency.col):
        start0 = i*k
        end0 = start0+k
        start1 = j*k
        end1 = start1+k
        sample = samples[start0:end0, start1:end1].todense()
        result += cp.trace(sample @ x[start0:end0,start1:end1].T)

    return result
    
def get_sdp_constraints(x: cp.Variable,
                        n: int,
                        k: int ) -> List[cp.Constraint]:
    constraints = [x >> 0] # Semi-definite positive

    # Diagonal blocks with identity
    eye = np.eye(k)
    for i in range(n):
        start = k*i
        end = start+k
        constraints.append(x[start:end, start:end] == eye)
        
    return constraints

def sdp_optimize_pairwise_matrix(samples: scipy.sparse.spmatrix,
                                 adjacency: scipy.sparse.coo_matrix,
                                 n: int,
                                 k: int,
                                 verbose: bool = False ) -> np.ndarray:
    x = cp.Variable(samples.shape, symmetric=True)
    objective_function = get_sdp_objective(
        x=x,
        samples=samples,
        adjacency=adjacency,
        k=k
    )
    constraints = get_sdp_constraints(x=x, n=n, k=k)
    
    problem = cp.Problem(cp.Maximize(objective_function), constraints)
    problem.solve(verbose=verbose)
    
    return x.value

def decompose_pairwise(pairwise: np.ndarray,
                       n: int,
                       k: int,
                       k2: int,
                       special: bool = False ) -> Tuple[np.ndarray, np.ndarray]:

    w, v = scipy.linalg.eigh(pairwise)
    
    # Compute the number of principal components
    v = v[:,-k2:]
    w = w[-k2:]
    
    v *= np.sqrt(w)
    v = v.reshape(n, k, k2)
    
    transforms = orthogonalize_matrices(v, special=special)
    
    return transforms, w

def sdp_ortho_group_synchronization(samples: scipy.sparse.csr_matrix,
                                    adjacency: scipy.sparse.coo_matrix,
                                    n: int,
                                    k: int,
                                    k2: int,
                                    special: bool = False,
                                    triu: bool = True,
                                    verbose: bool = False ) -> np.ndarray:
    if triu:
        adjacency = scipy.sparse.triu(adjacency)
    
    pairwise = sdp_optimize_pairwise_matrix(
        samples=samples,
        adjacency=adjacency,
        n=n,
        k=k,
        verbose=verbose
    )

    transforms, eigenvalues = decompose_pairwise(
        pairwise=pairwise, 
        n=n, 
        k=k, 
        k2=k2, 
        special=special
    )
    
    return transforms, eigenvalues

def bm_optimization_step(samples: scipy.sparse.csr_matrix,
                         adjacency: scipy.sparse.coo_matrix,
                         transforms: np.ndarray,
                         k: int,
                         special: bool = False) -> np.ndarray:
    result = np.zeros_like(transforms)

    for i, j in zip(adjacency.row, adjacency.col):
        start0 = i*k
        end0 = start0+k
        start1 = j*k
        end1 = start1+k
        sample = samples[start0:end0, start1:end1].todense()
        transform = transforms[j]
        result[i,:,:] += (sample @ transform)

    result = orthogonalize_matrices(result, special=special)
    return result

def bm_ortho_group_synchronization(samples: scipy.sparse.csr_matrix,
                                   adjacency: scipy.sparse.csr_matrix,
                                   n: int,
                                   k: int,
                                   k2: int,
                                   special: bool = False,
                                   verbose: bool = False,
                                   tol: float = 1e-6,
                                   max_iter: int = 256,
                                   start: Optional[np.ndarray] = None ) -> np.ndarray:
  x = start if start is not None else scipy.stats.ortho_group(k2).rvs(n)[:,:k,:]

  #last_cost = ortho_group_synchronization_objective(measurements, positions, x)
  for _ in range(max_iter):
    x = bm_optimization_step(
        samples=samples,
        adjacency=adjacency,
        transforms=x,
        k=k,
        special=special
    )

    if verbose:
        cost = ortho_group_synchronization_objective(
            samples=samples,
            adjacency=adjacency,
            transforms=x,
            k=k
        )
        print(cost, flush=True)
    
    #r = (cost - last_cost) / last_cost
    #if r < tol:
    #  break

  return x

def main(input_samples_path: str,
         input_adjacency_path: str,
         output_transforms_path: str,
         k: int,
         k2: int,
         method: str,
         output_pairwise_path: Optional[str] = None,
         output_eigenvalues_path: Optional[str] = None,
         special: bool = False,
         triu: bool = True,
         verbose: bool = False ):
    
    samples = scipy.sparse.load_npz(input_samples_path).tocsr(copy=False)
    adjacency = scipy.sparse.load_npz(input_adjacency_path).tocoo(copy=False)
    n = adjacency.shape[0] # TODO validate
    
    if method == 'sdp':
        transforms, eigenvalues = sdp_ortho_group_synchronization(
            samples=samples,
            adjacency=adjacency,
            n=n,
            k=k,
            k2=k2,
            special=special,
            triu=triu,
            verbose=verbose
        )
        
        if output_eigenvalues_path is not None:
            np.save(output_eigenvalues_path, eigenvalues)
        
    elif method == 'burer-monteiro':
        transforms = bm_ortho_group_synchronization(
            samples=samples,
            adjacency=adjacency,
            n=n,
            k=k,
            k2=k2,
            special=special,
            verbose=verbose
        )
        
    else:
        raise ValueError(
            'Unknown method, only sdp and burer-monteiro are supported'
        )
    
    if verbose:
        objective = ortho_group_synchronization_objective(
            samples=samples,
            adjacency=adjacency,
            transforms=transforms,
            k=k
        )
        print(f'Final objective function value: {objective}')
    
    if output_pairwise_path is not None:
        pairwise = calculate_pairwise_matrix(transforms, n=n, k=k, k2=k2)
        np.save(output_pairwise_path, pairwise)
    
    np.save(output_transforms_path, transforms)
    
if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(prog='Synchronize bases on a graph')
    parser.add_argument('-i', required=True)
    parser.add_argument('-a', '--adjacency', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('-k', required=True, type=int)
    parser.add_argument('-k2', type=int, required=True)
    parser.add_argument('-m', '--method', required=True)
    parser.add_argument('-p', '--pairwise')
    parser.add_argument('-e', '--eigenvalues')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--triangular_upper', action='store_true')
    parser.add_argument('--special', action='store_true')

    # Parse
    args = parser.parse_args()

    # Run the program
    main(
        input_samples_path=args.i,
        output_transforms_path=args.o,
        input_adjacency_path=args.adjacency,
        output_pairwise_path=args.pairwise,
        output_eigenvalues_path=args.eigenvalues,
        k=args.k,
        k2=args.k2,
        method=args.method,
        special=args.special,
        triu=args.triangular_upper,
        verbose=args.verbose
    )
    