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

from typing import Dict, List, Tuple

import argparse
import pickle
import numpy as np
import scipy.linalg
import cvxpy as cp

def get_optimization_objective_function(p: cp.Variable,
                                        graph: Dict[Tuple[int, int], np.ndarray],
                                        k: int) -> cp.Expression:
    objective_function = None
    for (idx0, idx1), cross_correlation in graph:
        start0 = idx0*k
        end0 = start0+k
        start1 = idx1*k
        end1 = start1+k

        trace = cp.trace(p[start0:end0,start1:end1].T @ cross_correlation)

        if objective_function is None:
            objective_function = trace
        else:
            objective_function = objective_function + trace
            
    return objective_function

def get_optimization_constraints(p: cp.Variable,
                                 n: int,
                                 k: int ) -> List[cp.Constraint]:
    constraints = [p >> 0]
    for i in range(n):
        start = k*i
        end = start+k
        constraints.append(p[start:end, start:end] == np.eye(k))
        
    return constraints

def optimize_pairwise_matrix(graph: Dict[Tuple[int, int], np.ndarray],
                             n: int, 
                             k: int ) -> np.ndarray:
    p = cp.Variable((n*k, )*2, symmetric=True)
    objective_function = get_optimization_objective_function(
        p=p,
        graph=graph,
        k=k
    )
    constraints = get_optimization_constraints(
        p=p,
        n=n,
        k=k
    )
    
    problem = cp.Problem(cp.Maximize(objective_function), constraints)
    objective_value = problem.solve()
    
    return p.value, objective_value
        
def compute_bases(pairwise: np.ndarray,
                  n: int,
                  k: int ) -> np.ndarray:
    w, v = scipy.linalg.eigh(pairwise, subset_by_index=[k*n-k, k*n-1])
    v *= np.sqrt(n)
    v = v.reshape(n, k, k)
    
    return v

def main(input_graph_path: str,
         output_bases: str,
         n: int,
         k: int ):
    
    graph = None
    with open(input_graph_path, 'rb') as f:
        graph = pickle.load(f)

    pairwise = optimize_pairwise_matrix(graph=graph, n=n, k=k)
    bases = compute_bases(pairwise=pairwise, n=n, k=k)
    
    np.save(output_bases, bases)
    
if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(prog='Synchronize bases on a graph')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('-n', required=True)
    parser.add_argument('-k', required=True)

    # Parse
    args = parser.parse_args()

    # Run the program
    main(
        input_graph_path=args.i,
        output_bases=args.o,
        n=args.n,
        k=args.k
    )
    