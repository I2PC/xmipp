#!/usr/bin/env python3
""""
**************************************************************************
*
* Authors:  David Herreros Calero (dherreros@cnb.csic.es)
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
*  e-mail address 'scipion@cnb.csic.es'
*
**************************************************************************
"""""

import math
import sys

import numpy as np
from scipy.spatial import KDTree

from pwem.convert import AtomicStructHandler

from xmipp_base import XmippScript


class ScriptAngularDensityAlign(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Angular Density Alignment: Rigid Alignemnt of point clouds and atomic structures')
        ## params
        self.addParamsLine('-i     <inputFile>         : Structure to be aligned.')
        self.addParamsLine('-r     <referenceFile>     : Reference atomic structure.')
        self.addParamsLine('[ --o    <outputFile> ]    : Aligned structure. If not provided, '
                           'the input structure will be overwritten.')
        self.addParamsLine('[ --rmax <radius> ]        : Maximum contact distance for neighbour '
                           'search. If not provided, several contact radii will be used.')
        self.addParamsLine('[ --nn <neighbours> ]      : Number of neighbours to search in KDTree.')
        ## examples
        self.addExampleLine('xmipp_pdb_angular_density_align -i path/to/structure.pdb -r path/to/reference.pdb')

    def run(self):
        inputStructure = self.getParam('-i')
        referenceStructure = self.getParam('-r')
        if self.checkParam('--o'):
            alignedStructure = self.getParam('--o')
        else:
            alignedStructure = inputStructure
        if self.checkParam('--rmax'):
            contactRadius_list = [self.getDoubleParam('--rmax')]
        else:
            contactRadius_list = [None]  # In Angstrom
        if self.checkParam('--nn'):
            size = self.getDoubleParam('--nn')
        else:
            size = 50
        self.alignStructure(inputStructure, referenceStructure, alignedStructure, contactRadius_list, size)

    def alignStructure(self, inputStructure, referenceStructure, alignedStructure, contactRadius_list, size):
        # Input Params
        bins = 11  # Number of bins in SPFH

        # Get Coordinates
        input_atoms = self.retrieveAtomCoords(inputStructure)
        ref_atoms = self.retrieveAtomCoords(referenceStructure)

        # Center Coordinates with origin
        center_input = self.createTransformation(np.eye(3), -np.mean(input_atoms[:, :3], axis=0))
        center_ref = self.createTransformation(np.eye(3), -np.mean(ref_atoms[:, :3], axis=0))
        center_with_ref = self.createTransformation(np.eye(3), np.mean(ref_atoms[:, :3], axis=0))
        input_atoms = np.transpose(np.dot(center_input, input_atoms.T))[:, :3]
        ref_atoms = np.transpose(np.dot(center_ref, ref_atoms.T))[:, :3]

        # Get KDTrees
        refTree = KDTree(ref_atoms)
        inputTree = KDTree(input_atoms)

        # Check if radius is provided
        if contactRadius_list[0] is None:
            contactRadius_list = [8, 12, 24]  # Try most common contact radius (better convergence but slower)

        # Main contact radius loop
        overall_cost = sys.maxsize
        size = 4
        while size <= 50:
            size += 1
            for contactRadius in contactRadius_list:

                # Initial neighbour search (neighobur distances + indeces)
                refDist, refIndeces = refTree.query(ref_atoms, k=size)
                inputDist, inputIndeces = inputTree.query(input_atoms, k=size)

                # Filter neighbours by distance and compute density vectors
                inputIndeces = self.filterNeighbours(inputIndeces, inputDist, contactRadius)
                refIndeces = self.filterNeighbours(refIndeces, refDist, contactRadius)
                vectors_input, filtered_input, inputDelete = self.computeDensityVectors(input_atoms, inputIndeces)
                vectors_ref, filtered_ref, refDelete = self.computeDensityVectors(ref_atoms, refIndeces)

                # Now we need more neighbours to compute and appropiate histogram
                refDist, refIndeces = refTree.query(ref_atoms, k=size)
                inputDist, inputIndeces = inputTree.query(input_atoms, k=size)

                # Filter neighbours by distance
                inputIndeces = self.filterNeighbours(inputIndeces, inputDist, contactRadius)
                refIndeces = self.filterNeighbours(refIndeces, refDist, contactRadius)

                # Update indexes according to density vectors
                inputIndeces = self.updateIndex(inputIndeces, inputDelete)
                refIndeces = self.updateIndex(refIndeces, refDelete)

                # Compute SPFH
                spfh_input, filtered_input, vectors_input, inputIndeces, inputDelete = self.SPFH(filtered_input,
                                                                                                vectors_input, bins,
                                                                                                inputIndeces)
                spfh_ref, filtered_ref, vectors_ref, refIndeces, refDelete = self.SPFH(filtered_ref, vectors_ref, bins,
                                                                                    refIndeces)

                # Update indexes according to density vectors
                inputIndeces = self.updateIndex(inputIndeces, inputDelete, onlyUpdate=True)
                refIndeces = self.updateIndex(refIndeces, refDelete, onlyUpdate=True)

                # Compute FPFH
                f_input = np.asarray(self.FPFH(filtered_input, spfh_input, inputIndeces))
                f_ref = np.asarray(self.FPFH(filtered_ref, spfh_ref, refIndeces))

                # Find correspondences in both directions
                correspondence_s_t, correspondence_t_s = self.findCorrespondences(f_input, f_ref)

                # Get only matching correspondences
                correspondence = []
                for idp in range(len(correspondence_s_t)):
                    pair = correspondence_s_t[idp]
                    if pair[0] == correspondence_t_s[pair[1]][1]:
                        correspondence.append(pair)

                p = 0
                l = 0
                while l <= 2:
                    # Get 1 percentile
                    p += 1
                    cost = np.asarray([pair[2] for pair in correspondence])
                    percentile = np.percentile(cost, p)

                    # Get point clouds using correspondences
                    pdb_matching = []
                    aux = []
                    chain_matching = []
                    for pair in correspondence:
                        if pair[2] < percentile:
                            pdb_matching.append(vectors_ref[pair[1]])
                            aux.append(filtered_ref[pair[1]])
                            chain_matching.append(vectors_input[pair[0]])
                    pdb_matching = np.asarray(pdb_matching)
                    aux = np.asarray(aux)
                    chain_matching = np.asarray(chain_matching)

                    l = len(chain_matching)

                # Create final transoformation due to correspondences
                center = self.createTransformation(np.eye(3), np.mean(aux, axis=0))
                M = self.superimposition_matrix(chain_matching.T, pdb_matching.T)
                Tr = center @ M

                # ICP Refinement of previous alignment
                transformed_cloud = input_atoms
                refTree = KDTree(ref_atoms)
                for _ in range(100):
                    X0 = np.ones((transformed_cloud.shape[0], 1))
                    transformed_cloud_step = np.hstack([transformed_cloud, X0])
                    transformed_cloud_step = np.transpose(Tr @ transformed_cloud_step.T)[:, :3]
                    _, indeces = refTree.query(transformed_cloud_step, k=1)
                    neighbours = ref_atoms[indeces]
                    M = self.superimposition_matrix(transformed_cloud_step.T, neighbours.T)
                    Tr = M @ Tr

                # Final transformation matrix
                Tr_trial = center_with_ref @ Tr @ center_input

                # Get trial cost (bidirectional RMSD based on KDTree)
                X0 = np.ones((input_atoms.shape[0], 1))
                transformed_input_atoms = np.hstack([input_atoms, X0])
                transformed_input_atoms = np.transpose(Tr @ transformed_input_atoms.T)[:, :3]

                _, indeces = refTree.query(transformed_input_atoms, k=1)
                neighbours_input_in_reference = ref_atoms[indeces]
                rmsd_input_in_reference = np.sqrt(np.mean((transformed_input_atoms - neighbours_input_in_reference) ** 2))

                transformed_input_tree = KDTree(transformed_input_atoms)
                _, indeces = transformed_input_tree.query(ref_atoms, k=1)
                neighbours_reference_in_input = transformed_input_atoms[indeces]
                rmsd_reference_in_input = np.sqrt(np.mean((ref_atoms - neighbours_reference_in_input) ** 2))

                # trial_cost = 0.5 * (rmsd_input_in_reference + rmsd_reference_in_input)
                trial_cost = rmsd_input_in_reference

                if trial_cost < overall_cost:
                    Tr_final = Tr_trial
                    overall_cost = trial_cost

        # Save results

        # Old implementation (Scipion not required)
        # input_atoms = self.retrieveAtomCoords(inputStructure)
        # aligned_input_coords = np.transpose(Tr_final @ input_atoms.T)[:, :3]
        # self.saveStructure(inputStructure, alignedStructure, aligned_input_coords)

        # New implementation (Scipion required)
        self.saveStructure(inputStructure, alignedStructure, Tr_final)

    # ------------------------------- UTILITIES -------------------------------
    ## Old implementation: This can be used if scipion python is not avalaible
    # def readPDB(self, structFile):
    #     with open(structFile) as f:
    #         lines = f.readlines()
    #     return lines

    ## Old implementation: This can be used if scipion python is not avalaible
    # def retrieveAtomCoords(self, structFile):
    #     lines = self.readPDB(structFile)
    #     coords = []
    #     for line in lines:
    #         if line.startswith("ATOM "):
    #             try:
    #                 coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54]), 1])
    #             except:
    #                 pass
    #     return np.asarray(coords)

    # New implementation: Scipion required
    def retrieveAtomCoords(self, structureFile):
        ah = AtomicStructHandler()
        ah.read(structureFile)
        atomIterator = ah.getStructure().get_atoms()
        coordAtoms = []
        for atom in atomIterator:
            # We only need to get the backbone to simplify the alignment
            if "C" == atom.get_name() or "N" == atom.get_name() or "O" == atom.get_name() or "CA" == atom.get_name():
                coordAtoms.append(np.append(atom.get_coord(), 1))
        return np.asarray(coordAtoms)

    ## Old implementation: This can be used if scipion python is not avalaible
    # def saveStructure(self, inputFile, structFile, coords):
    #     lines = self.readPDB(inputFile)
    #     newLines = []
    #     for ida, line in enumerate(lines):
    #         if line.startswith("ATOM "):
    #             try:
    #                 x = coords[ida, 0]
    #                 y = coords[ida, 1]
    #                 z = coords[ida, 2]
    #                 newLine = line[0:30] + "%8.3f%8.3f%8.3f" % (x, y, z) + line[54:]
    #             except:
    #                 pass
    #         else:
    #             newLine = line
    #         newLines.append(newLine)
    #
    #     with open(structFile, "w") as f:
    #         for line in newLines:
    #             f.write("%s" % line)

    # New implementation: Scipion required
    def saveStructure(self, inputFile, outputFile, Tranformation_Matrix):
        ah = AtomicStructHandler()
        ah.read(inputFile)
        ah.transform(Tranformation_Matrix)
        ah.writeAsPdb(outputFile)

    def createTransformation(self, R, t):
        tr = np.eye(4)
        tr[:3, :3] = R
        tr[:3, 3] = t
        return tr

    def SPFH(self, atoms, normals_atoms, bins, neighbours):
        s = np.zeros((1, 3 * bins))
        s_atoms = []
        s_vectors = []
        s_indeces = []
        delete = []
        for idp in range(atoms.shape[0]):
            points_index = neighbours[idp]
            alpha_vec = []
            gamma_vec = []
            theta_vec = []
            num_targets = len(points_index) - 1
            if num_targets <= 0:
                delete.append(neighbours[idp][0])
                continue
            normals = normals_atoms[points_index, :]
            points = atoms[points_index, :]
            s_atoms.append(points[0, :])
            s_vectors.append(normals[0, :])
            s_indeces.append(points_index)
            u = np.zeros((num_targets, 3))
            for idx in range(num_targets):
                angle_s = np.dot(normals[0], points[idx + 1] - points[0])
                angle_t = np.dot(normals[idx + 1], points[0] - points[idx + 1])
                if angle_s > angle_t:
                    u[idx] = normals[0]
                else:
                    u[idx] = normals[idx + 1]
            v = np.cross(u, points[1:] - points[0])
            w = np.cross(u, v)
            for idx in range(num_targets):
                alpha = np.dot(v[idx], normals[0])
                gamma = np.dot(u[idx], (points[idx + 1] - points[0]) / (np.linalg.norm(points[0] - points[idx + 1])))
                theta = math.atan2(np.dot(w[idx], normals[idx + 1]), np.dot(u[idx], normals[idx + 1]))
                alpha_vec.append(alpha)
                gamma_vec.append(gamma)
                theta_vec.append(theta)
            alpha_vec, _ = np.histogram(np.asarray(alpha_vec), bins=bins)
            gamma_vec, _ = np.histogram(np.asarray(gamma_vec), bins=bins)
            theta_vec, _ = np.histogram(np.asarray(theta_vec), bins=bins)
            aux = np.hstack((alpha_vec / np.sum(alpha_vec),
                             gamma_vec / np.sum(gamma_vec),
                             theta_vec / np.sum(theta_vec)))
            s = np.vstack((s, aux.reshape(1, -1)))
        s = s[1:, :]
        return s, np.asarray(s_atoms), np.asarray(s_vectors), s_indeces, delete

    def FPFH(self, atoms, s, neighbours):
        f = []
        for idp in range(len(atoms)):
            f_aux = s[idp]
            for idx in range(1, len(neighbours[idp])):
                f_aux = f_aux + s[neighbours[idp][idx]] / np.linalg.norm(atoms[idp] - atoms[neighbours[idp][idx]])
            f_aux = f_aux / (len(neighbours[idp]) - 1) + (len(neighbours[idp]) - 1) * s[idp] / len(neighbours[idp])
            f.append(f_aux)
        return np.asarray(f)

    def findCorrespondences(self, f_in, f_ref):
        dist_col, min_col = KDTree(f_ref).query(f_in, 1)
        dist_row, min_row = KDTree(f_in).query(f_ref, 1)
        correspondence_in_ref = [(id_in, min_col[id_in], dist_col[id_in])
                                 for id_in in range(f_in.shape[0])]
        correspondence_ref_in = [(id_in, min_row[id_in], dist_row[id_in])
                                 for id_in in range(f_ref.shape[0])]
        return correspondence_in_ref, correspondence_ref_in

    def computeDensityVectors(self, atoms, indeces):
        vectors = []
        out_atoms = []
        out_indices = []
        delete = []
        for index in indeces:
            points = atoms[index, :]
            if len(points) > 1:
                center_mass = np.mean(points[1:, :], axis=0)
                vector = center_mass - points[0]
                vector = vector / np.linalg.norm(vector)
                vectors.append(vector)
                out_atoms.append(points[0])
                out_indices.append(index)
            else:
                delete.append(index[0])
        return np.asarray(vectors), np.asarray(out_atoms), delete

    def computeNormals(self, atoms, indeces):
        normals = []
        out_atoms = []
        out_indices = []
        delete = []
        for index in indeces:
            points = atoms[index, :]
            if len(points) > 1:
                v1 = points[1] - points[0]
                v2 = points[2] - points[0]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                normals.append(normal)
                out_atoms.append(points[0])
                out_indices.append(index)
            else:
                delete.append(index[0])
        return np.asarray(normals), np.asarray(out_atoms), delete

    def filterNeighbours(self, neighbours, dist, Rmax):
        filtered = []
        for idx in range(len(neighbours)):
            val = neighbours[idx][dist[idx] <= Rmax]
            if not isinstance(val, list):
                filtered.append(val.tolist())
            else:
                filtered.append(val)
        return filtered

    def updateIndex(self, indeces, delete, onlyUpdate=False):
        # Remove sublist
        if not onlyUpdate:
            # Remove sublist
            copy_delete = delete
            for idd in range(len(copy_delete)):
                del_index = copy_delete[idd]
                indeces.remove(indeces[del_index])
                copy_delete = [idx - 1 if idx > del_index else idx for idx in copy_delete]

        # Remove indeces
        for del_index in sorted(delete):
            ids = 0
            while ids < len(indeces):
                idx = 0
                while idx < len(indeces[ids]):
                    if indeces[ids][idx] == del_index:
                        indeces[ids].remove(indeces[ids][idx])
                    idx += 1
                ids += 1

        # Update indexes
        delete = np.asarray(delete)
        for ids in range(len(indeces)):
            for idx in range(len(indeces[ids])):
                if np.any(indeces[ids][idx] > delete):
                    indeces[ids][idx] = indeces[ids][idx] - np.sum(indeces[ids][idx] > delete)
        return indeces

    # Rigid Alignment Methods
    def superimposition_matrix(self, v0, v1, scale=False):
        v0 = np.array(v0, dtype=np.float64, copy=False)[:3]
        v1 = np.array(v1, dtype=np.float64, copy=False)[:3]
        return self.affine_matrix_from_points(v0, v1, shear=False, scale=scale)

    def affine_matrix_from_points(self, v0, v1, shear=True, scale=True):
        v0 = np.array(v0, dtype=np.float64, copy=True)
        v1 = np.array(v1, dtype=np.float64, copy=True)

        ndims = v0.shape[0]
        if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
            raise ValueError("input arrays are of wrong shape or type")

        # move centroids to origin
        t0 = -np.mean(v0, axis=1)
        M0 = np.identity(ndims + 1)
        M0[:ndims, ndims] = t0
        v0 += t0.reshape(ndims, 1)
        t1 = -np.mean(v1, axis=1)
        M1 = np.identity(ndims + 1)
        M1[:ndims, ndims] = t1
        v1 += t1.reshape(ndims, 1)

        if shear:
            # Affine transformation
            A = np.concatenate((v0, v1), axis=0)
            u, s, vh = np.linalg.svd(A.T)
            vh = vh[:ndims].T
            B = vh[:ndims]
            C = vh[ndims:2 * ndims]
            t = np.dot(C, np.linalg.pinv(B))
            t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
            M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
        else:
            # Rigid transformation via SVD of covariance matrix
            u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
            # rotation matrix from SVD orthonormal bases
            R = np.dot(u, vh)
            if np.linalg.det(R) < 0.0:
                # R does not constitute right handed system
                R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
                s[-1] *= -1.0
            # homogeneous transformation matrix
            M = np.identity(ndims + 1)
            M[:ndims, :ndims] = R

        if scale and not shear:
            # Affine transformation; scale is ratio of RMS deviations from centroid
            v0 *= v0
            v1 *= v1
            M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

        # move centroids back
        M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
        M /= M[ndims, ndims]
        return M


if __name__ == "__main__":
    '''
    Use the Python installed with Scipion to run this script.

    Example:

        >>> scipion python 'which xmipp_pdb_angular_density_align' -i first.pdb -r second.pdb

    Other possibility is to call the script directly inside the environment of Scipion.

    Example (VirtualEnv)

        >>> source path/to/scipion/.scipion3env/bin/activate
        >>> (scipion) xmipp_pdb_angular_density_align -i first.pdb -r second.pdb

    Example (Conda)

        >>> conda activate YourScipionEnv
        >>> (scipion) xmipp_pdb_angular_density_align -i first.pdb -r second.pdb
    '''
    ScriptAngularDensityAlign().tryRun()
