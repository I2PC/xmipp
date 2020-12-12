/***************************************************************************
 *
 * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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
#ifndef _PROG_PSEUDOATOMS_SPH_DEFORM
#define _PROG_PSEUDOATOMS_SPH_DEFORM

#include <data/pdb.h>
#include <core/xmipp_program.h>
#include <core/metadata.h>
#include <core/matrix1d.h>
#include "core/xmipp_image.h"

#ifdef XOR
#undef XOR
#include <knn/kdtree_minkowski.h>
#include "core/xmipp_macros.h"
#endif

typedef Eigen::MatrixXd Matrix;
typedef knn::Matrixi Matrixi;

class ProgPseudoAtomsSphDeform: public XmippProgram
{
public:
    /** Input file */
    FileName fn_input;

    /** Reference file */
    FileName fn_ref;

    /** Output fileroot */
    FileName fn_out;

    /** Root name for several output files */
    FileName fn_root;

    /** Volume to deform */
	FileName fn_vol;

    /** Vector containing the deformation coefficients */
	Matrix1D<double> clnm;

    /** Copy of Optimizer steps */
    Matrix1D<double> steps_cp;

    /** Vector containing the degrees and Rmax of the basis */
	std::vector<double> basisParams;

    /** Atoms set */
    PDBRichPhantom Ai, Ar;

    /** Coords set */
    MultidimArray<double> Ci, Cr, Co;

    /** Eigen (software) Matrix */
    Matrix Er;

    /** Images */
	Image<double> Gx, Gy, Gz;

    /** Volumes */
	Image<double> V, Vo;

    /** Zernike and SPH coefficients vectors */
    Matrix1D<int> vL1, vN, vL2, vM;

    /** Deformation in pixels*/
	double deformation;

    /** Regularization */
    double lambda;

    /** Save the deformation of each voxel for local strain and rotation analysis */
    bool analyzeStrain;

    /** Radius optimization */
    bool optimizeRadius;

    /** Degree of Zernike polynomials and spherical harmonics */
    int L1, L2;

    /** Coefficient vector size */
	int vecSize;

    /** Maximum radius for the transformation */
	double Rmax;

    /** Save output volume */
	bool applyTransformation;

    /** Save the values of gx, gy and gz for local strain and rotation analysis */
	bool saveDeformation;

    /** Refine alignment of point cloud */\
    bool refineAlignment;

    /** KDTree for nearest neighbour search */
    knn::KDTreeMinkowski<double, knn::EuclideanDistance<double>> kdtree;

public:
    /** Params definitions */
    void defineParams();

    /** Read from a command line. */
    void readParams();

    /** Show parameters. */
    void show();

    /** Distance */
    double distance(double *pclnm);

    /** Run */
    void run();

    /** Fill degree and order vectors */
    void fillVectorTerms();

    /** Extract atom coordinates */
    void atoms2Coords(PDBRichPhantom &A, MultidimArray<double> &C);

    /** Convert MultidimArray to Eigen (software) matrix */
    void array2eigen(MultidimArray<double> &C, Matrix &E);

    /** Build KDTree */
    void buildTree(knn::KDTreeMinkowski<double, knn::EuclideanDistance<double>> &kdtree, Matrix &E);

    /** Compute center of mass of array */
    void massCenter(MultidimArray<double> &C, Matrix1D<double> &center);

    /** Inscribed radius search */
    void inscribedRadius(MultidimArray<double> &C, double &Rmax);

    /** Number of basis coefficients */
    void numCoefficients(int l1, int l2, int &vecSize);

    /** Vector positions to be minimized */
    void minimizepos(int L1, int l2, Matrix1D<double> &steps);

    /** Save vector to file */
    void writeVector(std::string outPath, Matrix1D<double> v, bool append);

    /** Apply deformation to volume */
    void deformVolume(Matrix1D<double> clnm);

    /** Compute strain */
    void computeStrain();
};
//@}
#endif
