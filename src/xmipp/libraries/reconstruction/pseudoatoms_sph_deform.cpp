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

#include "pseudoatoms_sph_deform.h"
#include <data/numerical_tools.h>
#include "data/fourier_filter.h"
#include <fstream>

void ProgPseudoAtomsSphDeform::defineParams()
{
	addUsageLine("Compute the deformation that properly fits two atoms set using spherical harmonics");
	addParamsLine("-a1 <file>               			  : First PDB to be deformed");
	addParamsLine("-a2 <file>               			  : Second PDB to be deformed");
	addParamsLine("  [-o <file=\"\">]          			  : Deformed atoms");
	addParamsLine("  [-v1 <volume=\"\">]          		  : Volume (of first PDB) to be deformed");
	addParamsLine("  [-v2 <volume=\"\">]          		  : Volume (of second PDB) to be deformed");
	addParamsLine("  [--oroot <rootname=\"Output\">]      : Root name for output files");
	addParamsLine("  [--analyzeStrain]     				  : Save the deformation of each voxel for local strain and rotation analysis");
	// addParamsLine("  [--optimizeRadius]    				  : Optimize the radius of each spherical harmonic");
	addParamsLine("  [--l1 <l1=3>]         				  : Degree Zernike Polynomials=1,2,3,...");
	addParamsLine("  [--l2 <l2=2>]         				  : Harmonical depth of the deformation=1,2,3,...");
	// addParamsLine("  [--regularization <l=0.00025>]       : Regularization weight");
	addParamsLine("  [--Rmax <r=-1>]       				  : Maximum radius for the transformation");
	addExampleLine("xmipp_pseudoatoms_sph_deform -i input.pdb -r ref.pdb -o input_deformed.pdb");
}

void ProgPseudoAtomsSphDeform::readParams()
{
	fn_input=getParam("-a1");
	fn_ref=getParam("-a2");
	fn_out=getParam("-o");
	fn_vol1 = getParam("-v1");
	fn_vol2 = getParam("-v2");
	fn_root = getParam("--oroot");
	if (fn_out=="")
		fn_out=fn_input;
	L1 = getIntParam("--l1");
	L2 = getIntParam("--l2");
	// lambda = getDoubleParam("--regularization");
	analyzeStrain=checkParam("--analyzeStrain");
	// optimizeRadius=checkParam("--optimizeRadius");
	Rmax = getDoubleParam("--Rmax");
	applyTransformation = false;
}

void ProgPseudoAtomsSphDeform::show()
{
	if (verbose==0)
		return;
	std::cout
	<< "Input atoms:          " << fn_input 	  << std::endl
	<< "Reference atoms:      " << fn_ref 		  << std::endl
	<< "Deformed atoms:       " << fn_out 		  << std::endl
	<< "Zernike Degree:       " << L1             << std::endl
	<< "SH Degree:            " << L2             << std::endl
	<< "Save deformation:     " << analyzeStrain  << std::endl
	;
}

double ProgPseudoAtomsSphDeform::distance(double *pclnm) {

	// Deformation for g_plus
	int l1,n,l2,m;
	double cost=0;
	size_t idxX0=0;
	size_t idxY0=VEC_XSIZE(clnm)/6;
	size_t idxZ0=2*idxY0;
	// double Ncount=0.0;
	double rmse_i=0.0;
	double rmse_o=0.0;
	double modg=0.0;
	double meanDistance;
	int k_nn;
	if (refineAlignment)
		k_nn = 1;
	else
		k_nn = 2;  //! 2 and 3 seems like a good option if no double comparison is perfomred per deformed-initial pair of volumes
	Matrixi indices;
    Matrix distances;
	Matrix queryPoint(3, 1);
	Matrix centerMass(3, 1);
	Matrix Eo_i(3, Ai.getNumberOfAtoms());
	FOR_ALL_ELEMENTS_IN_MATRIX1D(clnm)
		VEC_ELEM(clnm,i)=pclnm[i+1];
	for (size_t a=0; a<XSIZE(Ci); ++a) {
		double gx=0.0, gy=0.0, gz=0.0;
		double k = A2D_ELEM(Ci, 2, a);
		double i = A2D_ELEM(Ci, 1, a);
		double j = A2D_ELEM(Ci, 0, a);
		double Rmax2=Rmax*Rmax;
		double iRmax=1.0/Rmax;
		double k2=k*k;
		double kr=k*iRmax;
		double k2i2=k2+i*i;
		double ir=i*iRmax;
		double r2=k2i2+j*j;
		double jr=j*iRmax;
		double rr=std::sqrt(r2)*iRmax;
		for (size_t idx=idxX0; idx<idxY0; idx++) {
			if (VEC_ELEM(steps_cp,idx) == 1) {
				if (r2<Rmax2) {
					double zsph=0.0;
					l1 = VEC_ELEM(vL1,idx);
					n = VEC_ELEM(vN,idx);
					l2 = VEC_ELEM(vL2,idx);
					m = VEC_ELEM(vM,idx);
					zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
					if (rr>0 || l2==0) {
						gx += clnm[idx]        *(zsph);
						gy += clnm[idx+idxY0]  *(zsph);
						gz += clnm[idx+idxZ0]  *(zsph);
					}
				}
			}
		}
		if (saveDeformation) {
			Gx1(k,i,j)=gx;
			Gy1(k,i,j)=gy;
			Gz1(k,i,j)=gz;
		}
		if (applyTransformation) {
			RichAtom& atom_i=Ai.atomList[a];
			atom_i.x += gx;
			atom_i.y += gy;
			atom_i.z += gz;
		}
		// queryPoint(0,0) = j; queryPoint(1,0) = i; queryPoint(2,0) = k;
		queryPoint(0,0) = j+gx; queryPoint(1,0) = i+gy; queryPoint(2,0) = k+gz;
		Eo_i(0,a) = j+gx; Eo_i(1,a) = i+gy; Eo_i(2,a) = k+gz;
		A2D_ELEM(Co_i, 0, a) = j+gx; A2D_ELEM(Co_i, 1, a) = i+gy; A2D_ELEM(Co_i, 2, a) = k+gz;
		kdtree_r.query(queryPoint, k_nn, indices, distances);
		// meanDistance = 0.0;
		centerMass *= 0;
		for (size_t idp=0; idp<k_nn; idp++) {
			centerMass(0,0) += Er(0,indices(idp));
			centerMass(1,0) += Er(1,indices(idp));
			centerMass(2,0) += Er(2,indices(idp));
		}
		centerMass /= k_nn;
		meanDistance = pow(j+gx-centerMass(0,0),2) + pow(i+gy-centerMass(1,0),2) + pow(k+gz-centerMass(2,0),2);
		// meanDistance /= k_nn;
		rmse_i += meanDistance;
		modg+=gx*gx+gy*gy+gz*gz;
		// Ncount++;
	}
	// knn::KDTreeMinkowski<double, knn::EuclideanDistance<double>> kdtree_o_i;
	// buildTree(kdtree_o_i, Eo_i);
	// for (size_t a=0; a<XSIZE(Cr); ++a) {
	// 	double k = A2D_ELEM(Cr, 2, a);
	// 	double i = A2D_ELEM(Cr, 1, a);
	// 	double j = A2D_ELEM(Cr, 0, a);
	// 	queryPoint(0,0) = j; queryPoint(1,0) = i; queryPoint(2,0) = k;
	// 	kdtree_o_i.query(queryPoint, k_nn, indices, distances);
	// 	centerMass *= 0;
	// 	for (size_t idp=0; idp<k_nn; idp++) {
	// 		centerMass(0,0) += Eo_i(0,indices(idp));
	// 		centerMass(1,0) += Eo_i(1,indices(idp));
	// 		centerMass(2,0) += Eo_i(2,indices(idp));
	// 	}
	// 	centerMass /= k_nn;
	// 	meanDistance = pow(j-centerMass(0,0),2) + pow(i-centerMass(1,0),2) + pow(k-centerMass(2,0),2);
	// 	rmse_o += meanDistance;
	// }
	// if (applyTransformation)
	// 	Ai.write(fn_out);
		// Ai.write(fn_input.withoutExtension() + "_deformed.pdb");
	deformation_1_2=std::sqrt(modg/(XSIZE(Ci)));
	cost += 2*std::sqrt(rmse_i/XSIZE(Ci));
	// cost += 0.5*(std::sqrt(rmse_i/XSIZE(Ci)) + std::sqrt(rmse_o/XSIZE(Cr)));


	// Deformation for g_minus
	idxX0=3*VEC_XSIZE(clnm)/6;
	idxY0=4*VEC_XSIZE(clnm)/6;
	idxZ0=5*VEC_XSIZE(clnm)/6;
	// double Ncount=0.0;
	rmse_i=0.0;
	rmse_o=0.0;
	modg=0.0;
	Matrix Eo_r(3, Ar.getNumberOfAtoms());
	for (size_t a=0; a<XSIZE(Cr); ++a) {
		double gx=0.0, gy=0.0, gz=0.0;
		double k = A2D_ELEM(Cr, 2, a);
		double i = A2D_ELEM(Cr, 1, a);
		double j = A2D_ELEM(Cr, 0, a);
		double Rmax2=Rmax*Rmax;
		double iRmax=1.0/Rmax;
		double k2=k*k;
		double kr=k*iRmax;
		double k2i2=k2+i*i;
		double ir=i*iRmax;
		double r2=k2i2+j*j;
		double jr=j*iRmax;
		double rr=std::sqrt(r2)*iRmax;
		for (size_t idx=0; idx<VEC_XSIZE(clnm)/6; idx++) {
			if (VEC_ELEM(steps_cp,idx) == 1) {
				if (r2<Rmax2) {
					double zsph=0.0;
					l1 = VEC_ELEM(vL1,idx);
					n = VEC_ELEM(vN,idx);
					l2 = VEC_ELEM(vL2,idx);
					m = VEC_ELEM(vM,idx);
					zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
					if (rr>0 || l2==0) {
						gx += clnm[idx+idxX0]  *(zsph);
						gy += clnm[idx+idxY0]  *(zsph);
						gz += clnm[idx+idxZ0]  *(zsph);
					}
				}
			}
		}
		if (saveDeformation) {
			Gx2(k,i,j)=gx;
			Gy2(k,i,j)=gy;
			Gz2(k,i,j)=gz;
		}
		if (applyTransformation) {
			RichAtom& atom_i=Ar.atomList[a];
			atom_i.x += gx;
			atom_i.y += gy;
			atom_i.z += gz;
		}
		// queryPoint(0,0) = j; queryPoint(1,0) = i; queryPoint(2,0) = k;
		queryPoint(0,0) = j+gx; queryPoint(1,0) = i+gy; queryPoint(2,0) = k+gz;
		Eo_r(0,a) = j+gx; Eo_r(1,a) = i+gy; Eo_r(2,a) = k+gz;
		A2D_ELEM(Co_r, 0, a) = j+gx; A2D_ELEM(Co_r, 1, a) = i+gy; A2D_ELEM(Co_r, 2, a) = k+gz;
		kdtree_i.query(queryPoint, k_nn, indices, distances);
		// meanDistance = 0.0;
		centerMass *= 0;
		for (size_t idp=0; idp<k_nn; idp++) {
			centerMass(0,0) += Ei(0,indices(idp));
			centerMass(1,0) += Ei(1,indices(idp));
			centerMass(2,0) += Ei(2,indices(idp));
		}
		centerMass /= k_nn;
		meanDistance = pow(j+gx-centerMass(0,0),2) + pow(i+gy-centerMass(1,0),2) + pow(k+gz-centerMass(2,0),2);
		// meanDistance /= k_nn;
		rmse_i += meanDistance;
		modg+=gx*gx+gy*gy+gz*gz;
		// Ncount++;
	}
	// knn::KDTreeMinkowski<double, knn::EuclideanDistance<double>> kdtree_o_r;
	// buildTree(kdtree_o_r, Eo_r);
	// for (size_t a=0; a<XSIZE(Ci); ++a) {
	// 	double k = A2D_ELEM(Ci, 2, a);
	// 	double i = A2D_ELEM(Ci, 1, a);
	// 	double j = A2D_ELEM(Ci, 0, a);
	// 	queryPoint(0,0) = j; queryPoint(1,0) = i; queryPoint(2,0) = k;
	// 	kdtree_o_r.query(queryPoint, k_nn, indices, distances);
	// 	centerMass *= 0;
	// 	for (size_t idp=0; idp<k_nn; idp++) {
	// 		centerMass(0,0) += Eo_r(0,indices(idp));
	// 		centerMass(1,0) += Eo_r(1,indices(idp));
	// 		centerMass(2,0) += Eo_r(2,indices(idp));
	// 	}
	// 	centerMass /= k_nn;
	// 	meanDistance = pow(j-centerMass(0,0),2) + pow(i-centerMass(1,0),2) + pow(k-centerMass(2,0),2);
	// 	rmse_o += meanDistance;
	// }
	// if (applyTransformation)
	// 	Ar.write(fn_out);
		// Ar.write(fn_ref.withoutExtension() + "_deformed.pdb");
	deformation_2_1=std::sqrt(modg/(XSIZE(Cr)));
	cost += 2*std::sqrt(rmse_i/XSIZE(Cr));
	// cost += 0.5*(std::sqrt(rmse_i/XSIZE(Cr)) + std::sqrt(rmse_o/XSIZE(Ci)));

	// if (refineAlignment) {
		// Consistency g_plus
		idxX0=3*VEC_XSIZE(clnm)/6;
		idxY0=4*VEC_XSIZE(clnm)/6;
		idxZ0=5*VEC_XSIZE(clnm)/6;
		// double Ncount=0.0;
		double E_plus=0.0;
		for (size_t a=0; a<XSIZE(Co_i); ++a) {
			double gx=0.0, gy=0.0, gz=0.0;
			double k = A2D_ELEM(Co_i, 2, a);
			double i = A2D_ELEM(Co_i, 1, a);
			double j = A2D_ELEM(Co_i, 0, a);
			double Rmax2=Rmax*Rmax;
			double iRmax=1.0/Rmax;
			double k2=k*k;
			double kr=k*iRmax;
			double k2i2=k2+i*i;
			double ir=i*iRmax;
			double r2=k2i2+j*j;
			double jr=j*iRmax;
			double rr=std::sqrt(r2)*iRmax;
			for (size_t idx=0; idx<VEC_XSIZE(clnm)/6; idx++) {
				if (VEC_ELEM(steps_cp,idx) == 1) {
					if (r2<Rmax2) {
						double zsph=0.0;
						l1 = VEC_ELEM(vL1,idx);
						n = VEC_ELEM(vN,idx);
						l2 = VEC_ELEM(vL2,idx);
						m = VEC_ELEM(vM,idx);
						zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
						if (rr>0 || l2==0) {
							gx += clnm[idx+idxX0]  *(zsph);
							gy += clnm[idx+idxY0]  *(zsph);
							gz += clnm[idx+idxZ0]  *(zsph);
						}
					}
				}
			}
			E_plus += std::pow(A2D_ELEM(Ci, 0, a)-(j+gx), 2) + std::pow(A2D_ELEM(Ci, 1, a)-(i+gy), 2) + std::pow(A2D_ELEM(Ci, 2, a)-(k+gz), 2);
		}
		cost += 1*E_plus / XSIZE(Ci);


		// Consistency g_minus
		idxX0=0;
		idxY0=VEC_XSIZE(clnm)/6;
		idxZ0=2*idxY0;
		// double Ncount=0.0;
		double E_minus=0.0;
		for (size_t a=0; a<XSIZE(Co_r); ++a) {
			double gx=0.0, gy=0.0, gz=0.0;
			double k = A2D_ELEM(Co_r, 2, a);
			double i = A2D_ELEM(Co_r, 1, a);
			double j = A2D_ELEM(Co_r, 0, a);
			double Rmax2=Rmax*Rmax;
			double iRmax=1.0/Rmax;
			double k2=k*k;
			double kr=k*iRmax;
			double k2i2=k2+i*i;
			double ir=i*iRmax;
			double r2=k2i2+j*j;
			double jr=j*iRmax;
			double rr=std::sqrt(r2)*iRmax;
			for (size_t idx=idxX0; idx<idxY0; idx++) {
				if (VEC_ELEM(steps_cp,idx) == 1) {
					if (r2<Rmax2) {
						double zsph=0.0;
						l1 = VEC_ELEM(vL1,idx);
						n = VEC_ELEM(vN,idx);
						l2 = VEC_ELEM(vL2,idx);
						m = VEC_ELEM(vM,idx);
						zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
						if (rr>0 || l2==0) {
							gx += clnm[idx]        *(zsph);
							gy += clnm[idx+idxY0]  *(zsph);
							gz += clnm[idx+idxZ0]  *(zsph);
						}
					}
				}
			}
			E_minus += std::pow(A2D_ELEM(Cr, 0, a)-(j+gx), 2) + std::pow(A2D_ELEM(Cr, 1, a)-(i+gy), 2) + std::pow(A2D_ELEM(Cr, 2, a)-(k+gz), 2);
		}

		cost += 1*E_minus / XSIZE(Cr);
	// }


	// deformation=std::sqrt(modg/(Ncount));
	// return 0.5*(std::sqrt(rmse_i/XSIZE(Ci)) + std::sqrt(rmse_o/XSIZE(Cr))) + lambda*(deformation);
	return cost;
	// return std::sqrt(rmse/Ncount) + lambda*(deformation);
}

double atomsDeformSphGoal(double *p, void *vprm)
{
    ProgPseudoAtomsSphDeform *prm=(ProgPseudoAtomsSphDeform *) vprm;
	return prm->distance(p);
}

void ProgPseudoAtomsSphDeform::run() {

	// Preprocessing Steps
	saveDeformation=false;
	Ai.read(fn_input);
	Ar.read(fn_ref);
	atoms2Coords(Ai, Ci);
	atoms2Coords(Ar, Cr);
	// Matrix Er(3, Ar.getNumberOfAtoms());
	Er.resize(3, Ar.getNumberOfAtoms());
	array2eigen(Cr, Er);
	buildTree(kdtree_r, Er);
	Ei.resize(3, Ai.getNumberOfAtoms());
	array2eigen(Ci, Ei);
	buildTree(kdtree_i, Ei);
	Co_i.initZeros(3, Ai.getNumberOfAtoms());
	Co_r.initZeros(3, Ar.getNumberOfAtoms());
	if (Rmax<0) {
		double maxRi, maxRr;
		inscribedRadius(Ci, maxRi);
		inscribedRadius(Cr, maxRr);
		Rmax = std::max(maxRi, maxRr);
	}
	Matrix1D<double> steps, x;
	vecSize = 0;
	numCoefficients(L1,L2,vecSize);
	size_t totalSize = 3*vecSize;
	fillVectorTerms();
	// 2*totalSize to compute g_plus and g_minus
	clnm.initZeros(2*totalSize);
	x.initZeros(2*totalSize);

	// Initial alignment of point clouds
	refineAlignment = false;

	// Minimization Loop
	for (int h=0;h<=L2;h++) {
		steps.clear();
    	steps.initZeros(2*totalSize);
		minimizepos(L1,h,steps);
		steps_cp = steps;

		std::cout<<std::endl;
		std::cout<<"-------------------------- Basis Degrees: ("<<L1<<","<<h<<") --------------------------"<<std::endl;
		int iter;
        double fitness;
		powellOptimizer(x, 1, 2*totalSize, &atomsDeformSphGoal, this,
		                0.01, fitness, iter, steps, true);
		std::cout<<std::endl;
        std::cout << "Deformation S1 to S2: " << deformation_1_2 << std::endl;
		std::cout << "Deformation S2 to S1: " << deformation_2_1 << std::endl;
        // std::ofstream deformFile;
		// deformFile.open(fn_root+"_deformation.txt");
        // deformFile << deformation;
        // deformFile.close();
	}

	// Refine alignment of point cloud to approximate better to true neighbours
	// applyTransformation=true;
	// distance(x.adaptForNumericalRecipes());
	// applyTransformation=false;
	// atoms2Coords(Ai, Ci);
	refineAlignment = true;
	for (int h=0;h<=L2;h++) {
		// x.clear();
		// x.initZeros(totalSize);
		steps.clear();
    	steps.initZeros(2*totalSize);
		minimizepos(L1,h,steps);
		steps_cp = steps;
		int iter;
		double fitness;
		std::cout<<std::endl;
		std::cout<< "-------------------------- Refining Alignment: ("<<L1<<","<<h<<") --------------------------" << std::endl;
		powellOptimizer(x, 1, 2*totalSize, &atomsDeformSphGoal, this,
						0.01, fitness, iter, steps, true);
		std::cout<<std::endl;
		std::cout << "Deformation S1 to S2: " << deformation_1_2 << std::endl;
		std::cout << "Deformation S2 to S1: " << deformation_2_1 << std::endl;
		std::ofstream deformFile;
		deformFile.open(fn_root+"_deformation.txt");
		deformFile << "Deformation S1 to S2: " << deformation_1_2 << std::endl;
		deformFile << "Deformation S2 to S1: " << deformation_2_1 << std::endl;
		deformFile.close();
	}

	applyTransformation=true;
	Matrix1D<double> degrees;
	degrees.initZeros(3);
	VEC_ELEM(degrees,0) = L1;
	VEC_ELEM(degrees,1) = L2;
	VEC_ELEM(degrees,2) = Rmax;
	writeVector(fn_root+"_clnm.txt", degrees, false);
	writeVector(fn_root+"_clnm.txt", x, true);

	// Preprocessing needed to deform provided volume
	if (fn_vol1!="") {
		V1.read(fn_vol1);
		V1().setXmippOrigin();
		Vo1().initZeros(V1());
		Vo1().setXmippOrigin();
	}

	if (fn_vol2!="") {
		V2.read(fn_vol2);
		V2().setXmippOrigin();
		Vo2().initZeros(V2());
		Vo2().setXmippOrigin();
	}

	if (analyzeStrain)
    {
    	saveDeformation=true;
    	Gx1().initZeros(2*int(std::ceil(Rmax)), 2*int(std::ceil(Rmax)), 2*int(std::ceil(Rmax)));
    	Gy1().initZeros(Gx1());
    	Gz1().initZeros(Gx1());
    	Gx2().initZeros(Gx1());
    	Gy2().initZeros(Gx1());
    	Gz2().initZeros(Gx1());
		Gx1().setXmippOrigin();
    	Gy1().setXmippOrigin();
    	Gz1().setXmippOrigin();
    	Gx2().setXmippOrigin();
    	Gy2().setXmippOrigin();
    	Gz2().setXmippOrigin();
    }

	distance(x.adaptForNumericalRecipes()); // To save the output atoms
	Ai.write(fn_input.withoutExtension() + "_deformed.pdb");
	Ar.write(fn_ref.withoutExtension() + "_deformed.pdb");

	if (fn_vol1!="")
		deformVolume(x, V1, Vo1, "-", fn_vol1, Gx1, Gy1, Gz1); //Apply deformation to volume 1
	if (fn_vol2!="")
		deformVolume(x, V2, Vo2, "+", fn_vol2, Gx2, Gy2, Gz2); //Apply deformation to volume 2

	if (analyzeStrain) {
    	computeStrain(Gx1, Gy1, Gz1, fn_input);  //! In order to use xmipp_pdb_label_from_volume, origin should be moved to x,y,z Rmax,Rmax,Rmax
    	computeStrain(Gx2, Gy2, Gz2, fn_ref);  //! In order to use xmipp_pdb_label_from_volume, origin should be moved to x,y,z Rmax,Rmax,Rmax
	}
}

void ProgPseudoAtomsSphDeform::atoms2Coords(PDBRichPhantom &A, MultidimArray<double> &C) {
	C.initZeros(3, A.getNumberOfAtoms());
	for (size_t a=0; a<A.getNumberOfAtoms(); a++) {
		RichAtom& atom=A.atomList[a];
		A2D_ELEM(C, 0, a) = atom.x;
		A2D_ELEM(C, 1, a) = atom.y;
		A2D_ELEM(C, 2, a) = atom.z;
	}
}

void ProgPseudoAtomsSphDeform::array2eigen(MultidimArray<double> &C, Matrix &E) {
	for (size_t j=0; j<XSIZE(C); ++j) {
		E(0,j) = A2D_ELEM(C, 0, j);
		E(1,j) = A2D_ELEM(C, 1, j);
		E(2,j) = A2D_ELEM(C, 2, j);
	}
}

void ProgPseudoAtomsSphDeform::buildTree(knn::KDTreeMinkowski<double, knn::EuclideanDistance<double>> &kdtree, 
										 Matrix &E) {
	kdtree.setData(E);
	kdtree.setBucketSize(16);
	kdtree.setCompact(true);
	kdtree.setBalanced(true);
	kdtree.setSorted(true);
	kdtree.setTakeRoot(true);
	kdtree.setMaxDistance(0);
	kdtree.setThreads(1);
	// std::cout << std::endl;
	// std::cout << "Building Tree..." << std::endl;
	kdtree.build();
	// std::cout << "Done!" << std::endl;
}

void ProgPseudoAtomsSphDeform::massCenter(MultidimArray<double> &C, Matrix1D<double> &center) {
	center.initZeros(3);
	for (size_t j=0; j<XSIZE(C); ++j) {
		center[0] += A2D_ELEM(C, 0, j);
		center[1] += A2D_ELEM(C, 1, j);
		center[2] += A2D_ELEM(C, 2, j);
	}
	center[0] /= XSIZE(C);
	center[1] /= XSIZE(C);
	center[2] /= XSIZE(C);
}

void ProgPseudoAtomsSphDeform::inscribedRadius(MultidimArray<double> &C, double &Rmax) {
	Matrix1D<double> centerMass;
	double dist=0;
	// massCenter(C, centerMass);
	centerMass.initZeros(3);
	for (size_t j=0; j<XSIZE(C); ++j) {
		dist = sqrt(pow(centerMass[0]-A2D_ELEM(C, 0, j), 2.0) + 
					pow(centerMass[1]-A2D_ELEM(C, 1, j), 2.0) + 
					pow(centerMass[2]-A2D_ELEM(C, 2, j), 2.0));
		if (dist > Rmax)
			Rmax = dist;
	}
	Rmax += 0.1*Rmax;
}

void ProgPseudoAtomsSphDeform::numCoefficients(int l1, int l2, int &vecSize)
{
    for (int h=0; h<=l2; h++)
    {
        int numSPH = 2*h+1;
        int count=l1-h+1;
        int numEven=(count>>1)+(count&1 && !(h&1));
        if (h%2 == 0)
            vecSize += numSPH*numEven;
        else
        	vecSize += numSPH*(l1-h+1-numEven);
    }
}

void ProgPseudoAtomsSphDeform::fillVectorTerms()
{
    int idx = 0;
	vL1.initZeros(vecSize);
	vN.initZeros(vecSize);
	vL2.initZeros(vecSize);
	vM.initZeros(vecSize);
    for (int h=0; h<=L2; h++)
    {
        int totalSPH = 2*h+1;
        int aux = std::floor(totalSPH/2);
        for (int l=h; l<=L1; l+=2)
        {
            for (int m=0; m<totalSPH; m++)
            {
                VEC_ELEM(vL1,idx) = l;
                VEC_ELEM(vN,idx) = h;
                VEC_ELEM(vL2,idx) = h;
                VEC_ELEM(vM,idx) = m-aux;
                idx++;
            }
        }
    }
}

void ProgPseudoAtomsSphDeform::minimizepos(int L1, int l2, Matrix1D<double> &steps)
{
    int size = 0;
	numCoefficients(L1,l2,size);
    int totalSize = steps.size()/6;
    for (int idx=0; idx<size; idx++)
    {
		// g_plus
        VEC_ELEM(steps,idx) = 1;
        VEC_ELEM(steps,idx+totalSize) = 1;
        VEC_ELEM(steps,idx+2*totalSize) = 1;

		// g_minus
		VEC_ELEM(steps,idx+3*totalSize) = 1;
        VEC_ELEM(steps,idx+4*totalSize) = 1;
        VEC_ELEM(steps,idx+5*totalSize) = 1;
    }	
}

void ProgPseudoAtomsSphDeform::writeVector(std::string outPath, Matrix1D<double> v, bool append)
{
    std::ofstream outFile;
    if (append == false)
        outFile.open(outPath);
    else
        outFile.open(outPath, std::ios_base::app);
    FOR_ALL_ELEMENTS_IN_MATRIX1D(v)
        outFile << VEC_ELEM(v,i) << " ";
    outFile << std::endl;
}

void ProgPseudoAtomsSphDeform::deformVolume(Matrix1D<double> clnm, Image<double> &V, Image<double> &Vo,
											std::string mode, FileName fn_vol, Image<double> &Gx, Image<double> &Gy, 
											Image<double> &Gz) {
	const MultidimArray<double> &mV=V();
	double voxelI;
	size_t idxX0;
	size_t idxY0;
	size_t idxZ0;
	if (mode=="+"){
		idxX0=0;
		idxY0=VEC_XSIZE(clnm)/6;
		idxZ0=2*idxY0;
	}
	else {
		idxX0=3*VEC_XSIZE(clnm)/6;
		idxY0=4*VEC_XSIZE(clnm)/6;
		idxZ0=5*VEC_XSIZE(clnm)/6;
	}
	int l1,n,l2,m;
	for (int k=STARTINGZ(mV); k<=FINISHINGZ(mV); k++)	{
		for (int i=STARTINGY(mV); i<=FINISHINGY(mV); i++) {
			for (int j=STARTINGX(mV); j<=FINISHINGX(mV); j++) {
				double gx=0.0, gy=0.0, gz=0.0;
				double Rmax2=Rmax*Rmax;
				double iRmax=1.0/Rmax;
				double k2=k*k;
				double kr=k*iRmax;
				double k2i2=k2+i*i;
				double ir=i*iRmax;
				double r2=k2i2+j*j;
				double jr=j*iRmax;
				double rr=std::sqrt(r2)*iRmax;
				double zsph=0.0;
				for (size_t idx=0; idx<VEC_XSIZE(clnm)/6; idx++) {
					if (r2<Rmax2) {
                        l1 = VEC_ELEM(vL1,idx);
                        n = VEC_ELEM(vN,idx);
                        l2 = VEC_ELEM(vL2,idx);
                        m = VEC_ELEM(vM,idx);
                        zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
					}
                    // if (rr>0 || (l2==0 && l1==0))
					if (rr>0 || l2==0) {
						gx += VEC_ELEM(clnm,idx+idxX0)  *(zsph);
						gy += VEC_ELEM(clnm,idx+idxY0)  *(zsph);
						gz += VEC_ELEM(clnm,idx+idxZ0)  *(zsph);
					}
				}
				voxelI=mV.interpolatedElement3D(j+gx,i+gy,k+gz);
				Vo(k,i,j)=voxelI;

				// if (saveDeformation) {
				// 	Gx(k,i,j)=gx;
				// 	Gy(k,i,j)=gy;
				// 	Gz(k,i,j)=gz;
				// }
			}
		}
	}
	Vo.write(fn_vol.withoutExtension()+"_deformed_volume.vol");
}

// Number Spherical Harmonics ==============================================
#define Dx(V) (A3D_ELEM(V,k,i,jm2)-8*A3D_ELEM(V,k,i,jm1)+8*A3D_ELEM(V,k,i,jp1)-A3D_ELEM(V,k,i,jp2))/12.0
#define Dy(V) (A3D_ELEM(V,k,im2,j)-8*A3D_ELEM(V,k,im1,j)+8*A3D_ELEM(V,k,ip1,j)-A3D_ELEM(V,k,ip2,j))/12.0
#define Dz(V) (A3D_ELEM(V,km2,i,j)-8*A3D_ELEM(V,km1,i,j)+8*A3D_ELEM(V,kp1,i,j)-A3D_ELEM(V,kp2,i,j))/12.0

void ProgPseudoAtomsSphDeform::computeStrain(Image<double> &Gx, Image<double> &Gy, Image<double> &Gz, FileName fn_out)
{
	Image<double> LS, LR;
	LS().initZeros(Gx());
	LR().initZeros(Gx());

	// Gaussian filter of the derivatives
    FourierFilter f;
    f.FilterBand=LOWPASS;
    f.FilterShape=REALGAUSSIAN;
    f.w1=2;
    f.applyMaskSpace(Gx());
    f.applyMaskSpace(Gy());
    f.applyMaskSpace(Gz());

	Gx.write(fn_root+"_PPPGx.vol");
	Gy.write(fn_root+"_PPPGy.vol");
	Gz.write(fn_root+"_PPPGz.vol");

	MultidimArray<double> &mLS=LS();
	MultidimArray<double> &mLR=LR();
	MultidimArray<double> &mGx=Gx();
	MultidimArray<double> &mGy=Gy();
	MultidimArray<double> &mGz=Gz();
	Matrix2D<double> U(3,3), D(3,3), H(3,3);
	std::vector< std::complex<double> > eigs;
	H.initZeros();
	for (int k=STARTINGZ(mLS)+2; k<=FINISHINGZ(mLS)-2; ++k) {
		int km1=k-1;
		int kp1=k+1;
		int km2=k-2;
		int kp2=k+2;
		for (int i=STARTINGY(mLS)+2; i<=FINISHINGY(mLS)-2; ++i) {
			int im1=i-1;
			int ip1=i+1;
			int im2=i-2;
			int ip2=i+2;
			for (int j=STARTINGX(mLS)+2; j<=FINISHINGX(mLS)-2; ++j) {
				int jm1=j-1;
				int jp1=j+1;
				int jm2=j-2;
				int jp2=j+2;
				MAT_ELEM(U,0,0)=Dx(mGx); MAT_ELEM(U,0,1)=Dy(mGx); MAT_ELEM(U,0,2)=Dz(mGx);
				MAT_ELEM(U,1,0)=Dx(mGy); MAT_ELEM(U,1,1)=Dy(mGy); MAT_ELEM(U,1,2)=Dz(mGy);
				MAT_ELEM(U,2,0)=Dx(mGz); MAT_ELEM(U,2,1)=Dy(mGz); MAT_ELEM(U,2,2)=Dz(mGz);

				MAT_ELEM(D,0,0) = MAT_ELEM(U,0,0);
				MAT_ELEM(D,0,1) = MAT_ELEM(D,1,0) = 0.5*(MAT_ELEM(U,0,1)+MAT_ELEM(U,1,0));
				MAT_ELEM(D,0,2) = MAT_ELEM(D,2,0) = 0.5*(MAT_ELEM(U,0,2)+MAT_ELEM(U,2,0));
				MAT_ELEM(D,1,1) = MAT_ELEM(U,1,1);
				MAT_ELEM(D,1,2) = MAT_ELEM(D,2,1) = 0.5*(MAT_ELEM(U,1,2)+MAT_ELEM(U,2,1));
				MAT_ELEM(D,2,2) = MAT_ELEM(U,2,2);

				MAT_ELEM(H,0,1) = 0.5*(MAT_ELEM(U,0,1)-MAT_ELEM(U,1,0));
				MAT_ELEM(H,0,2) = 0.5*(MAT_ELEM(U,0,2)-MAT_ELEM(U,2,0));
				MAT_ELEM(H,1,2) = 0.5*(MAT_ELEM(U,1,2)-MAT_ELEM(U,2,1));
				MAT_ELEM(H,1,0) = -MAT_ELEM(H,0,1);
				MAT_ELEM(H,2,0) = -MAT_ELEM(H,0,2);
				MAT_ELEM(H,2,1) = -MAT_ELEM(H,1,2);

				A3D_ELEM(mLS,k,i,j)=fabs(D.det());
				allEigs(H,eigs);
				for (size_t n=0; n < eigs.size(); n++) {
					double imagabs=fabs(eigs[n].imag());
					if (imagabs>1e-6) {
						A3D_ELEM(mLR,k,i,j)=imagabs*180/PI;
						break;
					}
				}
			}
		}
	}
	LS.write(fn_out.withoutExtension()+"_strain.mrc");
	LR.write(fn_out.withoutExtension()+"_rotation.mrc");
}