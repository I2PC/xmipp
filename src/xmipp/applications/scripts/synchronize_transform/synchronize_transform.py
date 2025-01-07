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
                                          transforms: np.ndarray ) -> float:
    n, k, p = transforms.shape
    return np.trace(samples @ transforms.reshape(n*k, p))

def calculate_pairwise_matrix(transforms: np.ndarray) -> np.ndarray:
    n, k, p = transforms.shape
    p = transforms.reshape(n*k, p)
    return p @ p.T 

def get_sdp_objective(x: cp.Variable,
                      samples: scipy.sparse.csr_matrix ) -> cp.Expression:
    return cp.trace(samples @ x.T)
    
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
                                 n: int,
                                 k: int,
                                 verbose: bool = False ) -> np.ndarray:
    x = cp.Variable(samples.shape, symmetric=True)
    objective_function = get_sdp_objective(
        x=x,
        samples=samples
    )
    constraints = get_sdp_constraints(x=x, n=n, k=k)
    
    problem = cp.Problem(cp.Maximize(objective_function), constraints)
    problem.solve(verbose=verbose)
    
    return x.value

def decompose_pairwise(pairwise: np.ndarray,
                       n: int,
                       k: int ) -> Tuple[np.ndarray, np.ndarray]:

    w, v = np.linalg.eigh(pairwise)
    w = np.flip(w)
    v = np.flip(v, axis=1)
    
    v *= np.sqrt(w)
    transforms = v.reshape(n, k, k*n)

    return transforms, w

def decompose_bases(bases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n, k, p = bases.shape
    u, s, _ = np.linalg.svd(bases.reshape(n*k, p), full_matrices=False)

    u *= s
    transforms = np.reshape(u, (n, k, p))
    eigenvalues = np.square(s)

    return transforms, eigenvalues

def auto_trim_eigenvalues(eigenvalues: np.ndarray, power: float) -> int:
    rep = np.cumsum(eigenvalues)
    rep /= rep[-1]
    return np.where(rep >= power)

def sdp_ortho_group_synchronization(samples: scipy.sparse.csr_matrix,
                                    n: int,
                                    k: int,
                                    verbose: bool = False ) -> np.ndarray:
    pairwise = sdp_optimize_pairwise_matrix(
        samples=samples,
        n=n,
        k=k,
        verbose=verbose
    )

    transforms, eigenvalues = decompose_pairwise(
        pairwise=pairwise, 
        n=n, 
        k=k
    )
    
    return transforms, eigenvalues

def bm_optimization_step(samples: scipy.sparse.csr_matrix,
                         transforms: np.ndarray,
                         special: bool = False) -> np.ndarray:
    result = samples @ transforms.reshape(-1, transforms.shape[-1])
    result = result.reshape(transforms.shape)
    result = orthogonalize_matrices(result, special=special)
    return result

def bm_ortho_group_synchronization(samples: scipy.sparse.csr_matrix,
                                   n: int,
                                   k: int,
                                   special: bool = False,
                                   verbose: bool = False,
                                   tol: float = 1e-8,
                                   max_iter: int = 512,
                                   start: Optional[np.ndarray] = None ) -> np.ndarray:
    if start is None:
        p = 2*k+1
        start = np.random.randn(n*k, p)
        start, _ = np.linalg.qr(start, mode='reduced')
        start = start.reshape(n, k, p)
    
    x = start
    for _ in range(max_iter):
        prev_x = x
        x = bm_optimization_step(
            samples=samples,
            transforms=x,
            special=special
        )

        if verbose:
            cost = ortho_group_synchronization_objective(
                samples=samples,
                transforms=x
            )
            print(cost, flush=True)
        
        delta = np.linalg.norm(x-prev_x)
        if delta < tol:
            break

    return decompose_bases(x)

def main(input_samples_path: str,
         output_transforms_path: str,
         n: int,
         k: int,
         method: str,
         p: Optional[int] = None,
         group: Optional[str] = None,
         auto_trim: Optional[float] = None,
         output_pairwise_path: Optional[str] = None,
         output_eigenvalues_path: Optional[str] = None,
         verbose: bool = False ):
    
    samples = scipy.sparse.load_npz(input_samples_path).tocsr(copy=False)
    
    if method == 'sdp':
        transforms, eigenvalues = sdp_ortho_group_synchronization(
            samples=samples,
            n=n,
            k=k,
            verbose=verbose
        )
        
    elif method == 'burer-monteiro':
        transforms, eigenvalues = bm_ortho_group_synchronization(
            samples=samples,
            n=n,
            k=k,
            verbose=verbose
        )
        
    else:
        raise ValueError(
            'Unknown method, only sdp and burer-monteiro are supported'
        )
    
    if group is not None:
        p = k
    elif auto_trim is not None:
        p = auto_trim_eigenvalues(eigenvalues, auto_trim)
        
    if p is not None:
        transforms = transforms[:,:,:p]
        
    if group == 'O':
        transforms = orthogonalize_matrices(transforms, special=False)
    elif group == 'SO':
        transforms = orthogonalize_matrices(transforms, special=True)

    if verbose:
        objective = ortho_group_synchronization_objective(
            samples=samples,
            transforms=transforms
        )
        print(f'Final objective function value: {objective}')
    
    if output_pairwise_path is not None:
        pairwise = calculate_pairwise_matrix(transforms)
        np.save(output_pairwise_path, pairwise)

    if output_eigenvalues_path is not None:
        np.save(output_eigenvalues_path, eigenvalues)
    
    np.save(output_transforms_path, transforms)
    
if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(prog='Synchronize transforms on a graph')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('-n', required=True, type=int)
    parser.add_argument('-k', required=True, type=int)
    parser.add_argument('-p', type=int)
    parser.add_argument('--group')
    parser.add_argument('--auto_trim', type=float)
    parser.add_argument('--method', required=True)
    parser.add_argument('--pairwise')
    parser.add_argument('--eigenvalues')
    parser.add_argument('-v', '--verbose', action='store_true')

    # Parse
    args = parser.parse_args()

    # Run the program
    main(
        input_samples_path=args.i,
        output_transforms_path=args.o,
        output_pairwise_path=args.pairwise,
        output_eigenvalues_path=args.eigenvalues,
        n=args.n,
        k=args.k,
        p=args.p,
        group=args.group,
        auto_trim=args.auto_trim,
        method=args.method,
        verbose=args.verbose
    )
    