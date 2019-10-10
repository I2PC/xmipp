/***************************************************************************
 *
 * Authors:     Jeison Méndez García (jmendez@utp.edu.co)
 *
 * Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas -- IIMAS
 * Universidad Nacional Autónoma de México -UNAM
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

#include "angular_assignment_mag.h"

ProgAngularAssignmentMag::ProgAngularAssignmentMag() {
	produces_a_metadata = true;
	each_image_produces_an_output = false;
}

ProgAngularAssignmentMag::~ProgAngularAssignmentMag() {

}

void ProgAngularAssignmentMag::defineParams() {
	XmippMetadataProgram::defineParams();
	//usage
	addUsageLine( "Generates a list of candidates for angular assignment for each experimental image");
	addParamsLine("   -ref <md_file>             : Metadata file with input reference projections");
	addParamsLine("  [-odir <outputDir=\".\">]   : Output directory");
	addParamsLine("  [-sym <symfile=c1>]         : Enforce symmetry in projections");
	addParamsLine("  [-sampling <sampling=1.>]   : sampling");
	addParamsLine("  [-angleStep <angStep=3.>]   : angStep");
	addParamsLine("  [--maxShift <maxShift=-1.>]  : Maximum shift allowed (+-this amount)");
	addParamsLine("  [--Nsimultaneous <Nsim=1>]  : Nsimultaneous");
}

// Read arguments ==========================================================
void ProgAngularAssignmentMag::readParams() {
	XmippMetadataProgram::readParams();
	fnIn = XmippMetadataProgram::fn_in;
	fnOut = XmippMetadataProgram::fn_out;
	fnRef = getParam("-ref");
	fnDir = getParam("-odir");
	sampling = getDoubleParam("-sampling");
	angStep= getDoubleParam("-angleStep");
	XmippMetadataProgram::oroot = fnDir;
	fnSym = getParam("-sym");
	maxShift = getDoubleParam("--maxShift");
}

// Show ====================================================================
void ProgAngularAssignmentMag::show() {
	if (verbose > 0) {
		printf("%d reference images of %d x %d\n",int(sizeMdRef),int(Xdim),int(Ydim));
		printf("%d exp images of %d x %d in this group\n", int(sizeMdIn),int(Xdim), int(Ydim));
		printf("\nstartBand= %d\n", int(startBand));
		printf("finalBand= %d\n", int(finalBand));
		printf("n_bands= %d\n", int(n_bands));

		XmippMetadataProgram::show();
		std::cout << "Input references: " << fnRef << std::endl;
		std::cout << "Sampling: " << sampling << std::endl;
		std::cout << "Angular step: " << angStep << std::endl;
		std::cout << "Maximum shift: " << maxShift << std::endl;
	}
}

void ProgAngularAssignmentMag::startProcessing() {
	XmippMetadataProgram::startProcessing();
}

// compute variance respect to principal candidate
void neighVariance(Matrix1D<double> &neigh, double &retVal) {

	double val;
	double sum = 0.;
	double candVal = VEC_ELEM(neigh, 0); // value of candidate
	double diff = 0.; // difference between j and jp neighbors
	double N = neigh.vdim;
	for (int j = 0; j < N; ++j) {
		val = VEC_ELEM(neigh, j);
		diff = val - candVal;
		sum += diff * diff;
	}
	retVal = sum / N;
}

/*
 * In this method for each direction look for neighbors of size 3
 * nearest 2 neighbors to each direction
 * */
void ProgAngularAssignmentMag::computingNeighborGraph() {
	N_neighbors = 3; // including candidate itself
	std::vector<int> allNeighborsjp(sizeMdRef, 0); // for ordering
	std::vector<int> nearNeighbors(N_neighbors, 0);
	std::vector<double> vecNearNeighborsWeights(N_neighbors, 0.);
	Matrix1D<double> distanceToj;
	Matrix1D<double> dirj;
	Matrix1D<double> dirjp;
	Matrix1D<double> nearNeighborsDist;
	printf("processing neighbors graph...\n");
	FOR_ALL_OBJECTS_IN_METADATA(mdRef){
		double rotj;
		double tiltj;
		double psij;
		mdRef.getValue(MDL_ANGLE_ROT, rotj, __iter.objId);
		mdRef.getValue(MDL_ANGLE_TILT, tiltj, __iter.objId);
		mdRef.getValue(MDL_ANGLE_PSI, psij, __iter.objId);
		distanceToj.initZeros(sizeMdRef);
		nearNeighborsDist.initZeros(N_neighbors);
		Euler_direction(rotj, tiltj, psij, dirj);
		int jp = -1;
		for (MDIterator __iter2(mdRef); __iter2.hasNext(); __iter2.moveNext()) {
			jp += 1;
			double rotjp;
			double tiltjp;
			double psijp;
			mdRef.getValue(MDL_ANGLE_ROT, rotjp, __iter2.objId);
			mdRef.getValue(MDL_ANGLE_TILT, tiltjp, __iter2.objId);
			mdRef.getValue(MDL_ANGLE_PSI, psijp, __iter2.objId);

			//  check tilt value and if it is outside some range far from tiltj
			//      then don't make distance computation and put a big value, i.e. "possibly not neighbor"
			Euler_direction(rotjp, tiltjp, psijp, dirjp);
			VEC_ELEM(distanceToj,jp) = spherical_distance(dirj, dirjp);
			allNeighborsjp.at(jp) = jp;

			//  FALTA PONER SIMETRIA
		}

		//partial sort
		std::partial_sort(allNeighborsjp.begin(), allNeighborsjp.begin() + N_neighbors, allNeighborsjp.end(),
				[&distanceToj](int i, int j) {return VEC_ELEM(distanceToj,i) < VEC_ELEM(distanceToj,j);});

		for (int i = 0; i < N_neighbors; ++i) {
			nearNeighbors.at(i) = allNeighborsjp.at(i); //
			VEC_ELEM(nearNeighborsDist,i) = RAD2DEG(distanceToj[nearNeighbors[i]]);
		}

		double varAngDist = 0.;
		neighVariance(nearNeighborsDist, varAngDist);

		for (int i = 0; i < N_neighbors; ++i) {
			vecNearNeighborsWeights.at(i) = exp(-0.5 * VEC_ELEM(nearNeighborsDist, i) / varAngDist);
		}

		neighborsMatrix.push_back(nearNeighbors);
		neighborsWeights.push_back(vecNearNeighborsWeights);
	}
}

/*
 * In this method, for each direction, I look for neighbors within certain distance
 * */
void ProgAngularAssignmentMag::computingNeighborGraph2() {
	std::vector< std::vector<int> > allNeighborsjp;
	std::vector< std::vector<double> > allWeightsjp;
	Matrix1D<double> distanceToj;
	Matrix1D<double> dirj;
	Matrix1D<double> dirjp;
	double maxSphericalDistance=angStep*2.;
	printf("processing neighbors graph...\n");
/*comment
std::ofstream outCoord("/home/jeison/Escritorio/projCoordinates.txt"); // */
	FOR_ALL_OBJECTS_IN_METADATA(mdRef){
		double rotj;
		double tiltj;
		double psij;
/*comment
double cx,cy,cz;
mdRef.getValue(MDL_X,cx,__iter.objId);
mdRef.getValue(MDL_Y,cy,__iter.objId);
mdRef.getValue(MDL_Z,cz,__iter.objId); // */
		mdRef.getValue(MDL_ANGLE_ROT, rotj, __iter.objId);
		mdRef.getValue(MDL_ANGLE_TILT, tiltj, __iter.objId);
		mdRef.getValue(MDL_ANGLE_PSI, psij, __iter.objId);
		distanceToj.initZeros(sizeMdRef);
		Euler_direction(rotj, tiltj, psij, dirj);
		int jp = -1;
		std::vector<int> neighborsjp;
		std::vector<double> weightsjp;
		double thisSphericalDistance=0.;
		for (MDIterator __iter2(mdRef); __iter2.hasNext(); __iter2.moveNext()){
			jp += 1;
			double rotjp;
			double tiltjp;
			double psijp;
			mdRef.getValue(MDL_ANGLE_ROT, rotjp, __iter2.objId);
			mdRef.getValue(MDL_ANGLE_TILT, tiltjp, __iter2.objId);
			mdRef.getValue(MDL_ANGLE_PSI, psijp, __iter2.objId);
			Euler_direction(rotjp, tiltjp, psijp, dirjp);
			thisSphericalDistance=RAD2DEG(spherical_distance(dirj, dirjp)); // todo ask to COSS to modify spherical_distance() because belongs to xmippCore/core/

			if(thisSphericalDistance<maxSphericalDistance){
				neighborsjp.push_back(jp);
				double val=exp(- thisSphericalDistance/maxSphericalDistance );
				weightsjp.push_back(val);
			}
		}
		allNeighborsjp.push_back(neighborsjp);
		allWeightsjp.push_back(weightsjp);
/*comment
outCoord<<cx<<"\t"<<cy<<"\t"<<cz<<"\n"; // */

	} // END FOR_ALL_OBJECTS_IN_METADATA(mdRef)

/*comment
outCoord.close(); // */

	// compute Laplacian Matrix
	DMatrix L_mat;
	computeLaplacianMatrix(L_mat, allNeighborsjp,allWeightsjp);

/* time
double Inicio=std::clock();  // */
	// from diagSymMatrix3x3 method in resolution_directional.cpp
	std::cout<< "starts Eigen...\n";
	Matrix2D<double> B;
	B.resizeNoCopy(L_mat);
	B.initIdentity();
	generalizedEigs(L_mat, B, eigenvalues, eigenvectors);
	std::cout<< "finish\n";
/* time
double duration = ( std::clock() - Inicio ) / (double) CLOCKS_PER_SEC;
std::cout << "Operation of eigen-decomposition took "<< duration*1000 << "milliseconds" << std::endl; // */

/*comment print eigenvalues y eigenvectors
eigenvalues.write("/home/jeison/Escritorio/outEigenVal.txt");
eigenvectors.write("/home/jeison/Escritorio/outEigenVec.txt"); // */

}

/* Laplacian Matrix is basic for signal graph processing stage
 * is computed only once within preProcess() method */
void ProgAngularAssignmentMag::computeLaplacianMatrix(Matrix2D<double> &L,
		const std::vector< std::vector<int> > &allNeighborsjp,
		const std::vector< std::vector<double> > &allWeightsjp){

	L.initZeros(sizeMdRef,sizeMdRef);

	for(int i=0; i<sizeMdRef; ++i){
		std::vector<int> neighborsjp=allNeighborsjp[i];
		std::vector<double> weightsjp=allWeightsjp[i];
		double sumWeight=0.;
		int indx=0;
		int j=-1;
		for(std::vector<int>::iterator it=neighborsjp.begin(); it!=neighborsjp.end(); ++it){
			j+=1;
			indx=(*it);
			MAT_ELEM(L,i,indx)=-weightsjp[j];
			sumWeight+=weightsjp[j];
		}
		MAT_ELEM(L,i,i)=sumWeight-1.; // -1 because is necessary to remove the "central" weight
	}

/*comment write-out Laplacian matrix to check
L.write("/home/jeison/Escritorio/LaplacianMatrix.txt"); // */

}

void ProgAngularAssignmentMag::preProcess() {

	mdIn.read(fnIn);
	mdRef.read(fnRef);

	// size of images
	size_t Zdim;
	size_t Ndim;
	getImageSize(mdIn, Xdim, Ydim, Zdim, Ndim);

	// some constants
	n_rad = size_t(Xdim / 2.);

	startBand = size_t((sampling * Xdim) / 80.); // 100
	finalBand = size_t((sampling * Xdim) / (sampling * 3));

	n_bands = finalBand - startBand;

	n_ang = size_t(180);
	n_ang2 = 2 * n_ang;
	if (maxShift==-1.){
		maxShift = .10 * Xdim;
	}

	// read reference images
	FileName fnImgRef;
	MDRow rowRef;
	sizeMdRef = mdRef.size();

	// how many input images
	sizeMdIn = mdIn.size();

	// reference image related
	Image<double> ImgRef;
	MultidimArray<double> MDaRef(Ydim, Xdim);
	MultidimArray<std::complex<double> > MDaRefF;
	MultidimArray<std::complex<double> > MDaRefF2;
	MultidimArray<double> MDaRefFM;
	MultidimArray<double> MDaRefFMs;
	MultidimArray<double> MDaRefFMs_polarPart(n_bands, n_ang2);
	MultidimArray<std::complex<double> > MDaRefFMs_polarF;

	size_t first = 0;
	MultidimArray<double> refPolar(n_rad, n_ang2);
	MultidimArray<std::complex<double> > MDaRefAuxF;

	computeCircular(); //precompute circular mask

	// for storage of rot and tilt of reference images
	referenceRot.resize(sizeMdRef);
	referenceTilt.resize(sizeMdRef);

	// try to storage all data related to reference images in memory
	printf("processing reference library...\n");
	int j = -1;
/* time
double Inicio=std::clock(); // */
	FOR_ALL_OBJECTS_IN_METADATA(mdRef){
		j += 1;
		// reading image
		mdRef.getValue(MDL_IMAGE, fnImgRef, __iter.objId);
		ImgRef.read(fnImgRef);
		MDaRef = ImgRef();
		// store to call in processImage method
		double rot;
		double tilt;
		double psi;
		mdRef.getValue(MDL_ANGLE_ROT, rot, __iter.objId);
		mdRef.getValue(MDL_ANGLE_TILT, tilt, __iter.objId);
		mdRef.getValue(MDL_ANGLE_PSI, psi, __iter.objId);
		referenceRot.at(j) = rot;
		referenceTilt.at(j) = tilt;
		// processing reference image
		vecMDaRef.push_back(MDaRef);
		applyFourierImage2(MDaRef, MDaRefF);
		vecMDaRefF.push_back(MDaRefF);
		//fourier of polar image in real space
		refPolar = imToPolar(MDaRef, first, n_rad);
		applyFourierImage3(refPolar, MDaRefAuxF, n_ang);
		vecMDaRef_polarF.push_back(MDaRefAuxF);
		// fourier of polar magnitude spectra
		transformerImage.getCompleteFourier(MDaRefF2);
		getComplexMagnitude(MDaRefF2, MDaRefFM);
		completeFourierShift(MDaRefFM, MDaRefFMs);
/*comment
if(j==0){
MDaRefFM.write("/home/jeison/Escritorio/FourierMagnitude.txt");
MDaRefFMs.write("/home/jeison/Escritorio/FourierMagnitudeShifted.txt");
} // */
		MDaRefFMs_polarPart = imToPolar(MDaRefFMs, startBand, finalBand);
		applyFourierImage2(MDaRefFMs_polarPart, MDaRefFMs_polarF, n_ang);
		vecMDaRefFMs_polarF.push_back(MDaRefFMs_polarF);
	}

/* time
double duration = ( std::clock() - Inicio ) / (double) CLOCKS_PER_SEC;
std::cout << "Operation in preProcess took "<< duration*1000 << "milliseconds" << std::endl; // */

	mdOut.setComment("experiment for metadata output containing data for reconstruction");

	// Define the neighborhood graph and Laplacian Matrix
	computingNeighborGraph2();
}

/* Apply graph signal processing to cc-vector using the Laplacian eigen-decomposition
 * */
void ProgAngularAssignmentMag::graphFourierFilter(Matrix1D<double> &ccVecIn, Matrix1D<double> &ccVecOut ){
	// graph signal processing filtered iGFT
	Matrix1D<double> ccGFT;
	ccGFT.initZeros(sizeMdRef);
	std::vector<double> filt_iGFT_cc(sizeMdRef, 0);

	Matrix2D<double> eigenvectorTrans=eigenvectors.transpose();
	ccGFT=eigenvectorTrans*ccVecIn;
/*comment
ccGFT.write("/home/jeison/Escritorio/testGFT.txt"); // */

	// define filtering base
	int cutEig=(sizeMdRef>1000) ? int(.05 * sizeMdRef + 1) : int(.50 * sizeMdRef + 1);
	for(int k=cutEig; k<sizeMdRef; ++k){
		VEC_ELEM(ccGFT,k)=0.;
	}
/*comment
ccGFT.write("/home/jeison/Escritorio/testGFT_filt.txt"); // */

	// apply filter to ccvec
	ccVecOut=eigenvectors*ccGFT;

}

/*  this one gives a value for "spatial energy concentration"
 * I hope small values for high values energy concentration in only one region over the sphere
 * */
double ProgAngularAssignmentMag::energyDistribution(const Matrix1D<double> &dirj,
		const std::vector<unsigned int> &Idx, const std::vector<double> &ccVector,
		const std::vector<unsigned int> &candidates){

	double retVal=0;
	Matrix1D<double> dirk;
	for(int k=1; k<sizeMdRef; ++k){
		double rotk, tiltk, psik;
		rotk=referenceRot.at(Idx[k]);
		tiltk=referenceTilt.at(Idx[k]);
		psik=0.;
		Euler_direction(rotk, tiltk, psik, dirk);
		double sphericalDistance=RAD2DEG(spherical_distance(dirj, dirk));
		retVal += sphericalDistance*(1 - ccVector[Idx[k]]);
if(Idx[k] != candidates[Idx[k]])
	std::cout<<"Hey, they're different\n";
	}

	return retVal;
}

/* this one gives a value for "spatial energy concentration"
 * I hope small values for high values energy concentration in only one region over the sphere
 * */
double ProgAngularAssignmentMag::energyDistribution(const Matrix1D<double> &dirj,
		const std::vector<unsigned int> &Idx, const Matrix1D<double> &ccVector){

	double retVal=0;
	Matrix1D<double> dirk;
	for(int k=1; k<sizeMdRef; ++k){
		double rotk, tiltk, psik;
		rotk=referenceRot.at(Idx[k]);
		tiltk=referenceTilt.at(Idx[k]);
		psik=0.;
		Euler_direction(rotk, tiltk, psik, dirk);
		double sphericalDistance=RAD2DEG(spherical_distance(dirj, dirk));
		retVal += sphericalDistance*(1 - ccVector[Idx[k]]);
	}

	return retVal;
}

void ProgAngularAssignmentMag::processImage(const FileName &fnImg,const FileName &fnImgOut,
		const MDRow &rowIn, MDRow &rowOut) {
	// experimental image related
	rowOut = rowIn;

/* time
double Inicio=std::clock(); // */

	// input image related
	MDRow rowRef;
	Image<double> ImgIn;
	MultidimArray<double> MDaIn(Ydim, Xdim);
	MultidimArray<std::complex<double> > MDaInF;
	MultidimArray<std::complex<double> > MDaInF2;
	MultidimArray<double> MDaInFM;
	MultidimArray<double> MDaInFMs;
	MultidimArray<double> MDaInFMs_polarPart(n_bands, n_ang2);
	MultidimArray<std::complex<double> > MDaInFMs_polarF;

	// processing input image
	ImgIn.read(fnImg);
	MDaIn = ImgIn();
	applyFourierImage2(MDaIn, MDaInF);
	transformerImage.getCompleteFourier(MDaInF2);
	getComplexMagnitude(MDaInF2, MDaInFM);
	completeFourierShift(MDaInFM, MDaInFMs);
	MDaInFMs_polarPart = imToPolar(MDaInFMs, startBand, finalBand);
	applyFourierImage2(MDaInFMs_polarPart, MDaInFMs_polarF, n_ang);

	double psi;
	double cc_coeff;
	double Tx;
	double Ty;
	int maxAccepted = 8;

	std::vector<unsigned int> candidatesFirstLoop(sizeMdRef, 0);
	std::vector<unsigned int> Idx(sizeMdRef, 0);
	std::vector<double> candidatesFirstLoopCoeff(sizeMdRef, 0);

	Matrix1D<double> ccvec;
	ccvec.initZeros(sizeMdRef);
Matrix1D<double> txvec;
txvec.initZeros(sizeMdRef);
Matrix1D<double> tyvec;
tyvec.initZeros(sizeMdRef);
Matrix1D<double> psivec;
psivec.initZeros(sizeMdRef);

	std::vector<double> bestTx(sizeMdRef, 0);
	std::vector<double> bestTy(sizeMdRef, 0);
	std::vector<double> bestPsi(sizeMdRef, 0);

	MultidimArray<double> ccMatrixRot;
	MultidimArray<double> ccVectorRot;

	// loop over all reference images
	for (int k = 0; k < sizeMdRef; ++k) {
		// computing relative rotation and shift
		ccMatrix(MDaInFMs_polarF, vecMDaRefFMs_polarF[k], ccMatrixRot);
		maxByColumn(ccMatrixRot, ccVectorRot);
		peaksFound = 0;
		std::vector<double> cand(maxAccepted, 0.);
		rotCandidates3(ccVectorRot, cand, XSIZE(ccMatrixRot));
		bestCand(MDaIn, MDaInF, vecMDaRef[k], cand, psi, Tx, Ty, cc_coeff);

		// all results are storage for posterior partial_sort
		Idx[k] = k; // for sorting
		candidatesFirstLoop[k] = k; // for access in second loop
		candidatesFirstLoopCoeff[k] = cc_coeff;
		VEC_ELEM(ccvec,k)=cc_coeff;
		bestTx[k] = Tx; // todo if works then delete std-vectors and keep only Matrix1D
VEC_ELEM(txvec,k)= Tx;
		bestTy[k] = Ty;
VEC_ELEM(tyvec,k)= Ty;
		bestPsi[k] = psi;
VEC_ELEM(psivec,k) = psi;
	}

/*comment
ccvec.write("/home/jeison/Escritorio/testVectCC_1stLoop.txt");
txvec.write("/home/jeison/Escritorio/testVectTx_1stLoop.txt");
tyvec.write("/home/jeison/Escritorio/testVectTy_1stLoop.txt");
psivec.write("/home/jeison/Escritorio/testVectPsi_1stLoop.txt");
// */

//// ========graph filter to ccVect from first loop, using filtered signal================
////         to sort candidates previous to second loop processing
//
//// ================ Graph Filter Process =================
//	Matrix1D<double> ccvec_filt;
//	graphFourierFilter(ccvec,ccvec_filt);
//ccvec_filt.write("/home/jeison/Escritorio/test_ccVect_filt.txt");
//
//	// search rotation with polar real image representation over 10% of reference images
//	// nCand value should be 10% for experiments with C1 symmetry (more than 1000 references)
//	// but for I1 symmetry, for example, should be at least 50%.
//	int nCand = (sizeMdRef>1000) ? int(.10 * sizeMdRef + 1) : int(.50 * sizeMdRef + 1);
//
//std::cout<<"nCand: "<<nCand<<"\n";
//
////	//ordering using cross-corr coefficient values computed in first loop
////	std::partial_sort(Idx.begin(), Idx.begin()+nCand, Idx.end(),
////			[&candidatesFirstLoopCoeff](int i, int j) {return candidatesFirstLoopCoeff[i] > candidatesFirstLoopCoeff[j];});
//
//	// ordering using cross-corr coefficient values filtered using graph signal approach
//	std::partial_sort(Idx.begin(), Idx.begin()+nCand, Idx.end(),
//			[&ccvec_filt](int i, int j) {return ccvec_filt[i] > ccvec_filt[j];});
//
//	std::vector<unsigned int> candidatesSecondLoop(nCand, 0);
//	std::vector<unsigned int> Idx2(nCand, 0);
//	std::vector<double> candidatesSecondLoopCoeff(nCand, 0.);
//	std::vector<double> bestTx2(nCand, 0.);
//	std::vector<double> bestTy2(nCand, 0.);
//	std::vector<double> bestPsi2(nCand, 0.);
//
//	size_t first = 0;
//	MultidimArray<double> inPolar(n_rad, n_ang2);
//	MultidimArray<double> MDaExpShiftRot2; // transform experimental
//	MDaExpShiftRot2.setXmippOrigin();
//	MultidimArray<double> ccMatrixRot2;
//	MultidimArray<double> ccVectorRot2;
//	MultidimArray<std::complex<double> > MDaInAuxF;
//	for (int k = 0; k < nCand; ++k) {
//		//apply transform to experimental image
//		double rotVal = -1. * bestPsi[Idx[k]]; // -1. because I return parameters for reference image instead of experimental
//		double trasXval = -1. * bestTx[Idx[k]];
//		double trasYval = -1. * bestTy[Idx[k]];
//		_applyShiftAndRotation(MDaIn,rotVal,trasXval,trasYval,MDaExpShiftRot2);
//		//fourier experimental image
//		_applyFourierImage2(MDaExpShiftRot2, MDaInF);
//		// polar experimental image
//		inPolar = imToPolar(MDaExpShiftRot2, first, n_rad);
//		_applyFourierImage2(inPolar, MDaInAuxF, n_ang); // TODO check!!  at least first time, it uses the same transformer used in the previous loop which have a different size
//
//		// find rotation and shift
//		ccMatrix(MDaInAuxF,vecMDaRef_polarF[candidatesFirstLoop[Idx[k]]],ccMatrixRot2);
//		maxByColumn(ccMatrixRot2, ccVectorRot2);
//		peaksFound = 0;
//		std::vector<double> cand(maxAccepted, 0.);
//		rotCandidates3(ccVectorRot2, cand, XSIZE(ccMatrixRot2));
//		bestCand(MDaExpShiftRot2, MDaInF, vecMDaRef[Idx[k]], cand, psi,Tx, Ty, cc_coeff); //
//
//		// if its better and shifts are within range then update
//		double testShiftTx = bestTx[Idx[k]] + Tx;
//		double testShiftTy = bestTy[Idx[k]] + Ty;
//		if (cc_coeff > candidatesFirstLoopCoeff[Idx[k]]
//				&& std::abs(testShiftTx) < maxShift	&& std::abs(testShiftTy) < maxShift) {
//			Idx2[k] = k;
//			candidatesSecondLoop[k] = candidatesFirstLoop[Idx[k]];
//			candidatesSecondLoopCoeff[k] = cc_coeff;
//			bestTx2[k] = testShiftTx;
//			bestTy2[k] = testShiftTy;
//			bestPsi2[k] = bestPsi[Idx[k]] + psi;
//		} else {
//			Idx2[k] = k;
//			candidatesSecondLoop[k] = candidatesFirstLoop[Idx[k]];
//			candidatesSecondLoopCoeff[k] =candidatesFirstLoopCoeff[Idx[k]];
//			bestTx2[k] = bestTx[Idx[k]];
//			bestTy2[k] = bestTy[Idx[k]];
//			bestPsi2[k] = bestPsi[Idx[k]];
//		}
//	}

// ==========graph filter processing after second loop=========================
	//  second loop, "real search" over percentage of references
	//  then ccVect after second loop must have sizeMdRef elements

	// search rotation with polar real image representation over 10% of reference images
	// nCand value should be 10% for experiments with C1 symmetry (more than 1000 references)
	// but for I1 symmetry, for example, should be at least 50%.
	int nCand = (sizeMdRef>1000) ? int(.10 * sizeMdRef + 1) : int(.50 * sizeMdRef + 1);

	// ordering using cross-corr coefficient values computed in first loop
	// only best reference directions should be refined with polar real-space alignment
	std::partial_sort(Idx.begin(), Idx.begin()+nCand, Idx.end(),
			[&candidatesFirstLoopCoeff](int i, int j) {return candidatesFirstLoopCoeff[i] > candidatesFirstLoopCoeff[j];});

	// same size as in first loop
	std::vector<unsigned int> candidatesSecondLoop(sizeMdRef, 0);
	std::vector<unsigned int> Idx2(sizeMdRef, 0);
	std::vector<unsigned int> Idx3(sizeMdRef, 0);
	std::vector<double> candidatesSecondLoopCoeff(sizeMdRef, 0.);
	std::vector<double> bestTx2(sizeMdRef, 0.);
	std::vector<double> bestTy2(sizeMdRef, 0.);
	std::vector<double> bestPsi2(sizeMdRef, 0.);

	size_t first = 0;
	MultidimArray<double> inPolar(n_rad, n_ang2);
	MultidimArray<double> MDaExpShiftRot2; // transform experimental
	MDaExpShiftRot2.setXmippOrigin();
	MultidimArray<double> ccMatrixRot2;
	MultidimArray<double> ccVectorRot2;
	MultidimArray<std::complex<double> > MDaInAuxF;
	for(int k = 0; k < sizeMdRef; ++k) {
		if(k<nCand){
			//apply transform to experimental image
			double rotVal = -1. * bestPsi[Idx[k]]; // -1. because I return parameters for reference image instead of experimental
			double trasXval = -1. * bestTx[Idx[k]];
			double trasYval = -1. * bestTy[Idx[k]];
			applyShiftAndRotation(MDaIn,rotVal,trasXval,trasYval,MDaExpShiftRot2);
			//Fourier of transformed experimental image
			applyFourierImage2(MDaExpShiftRot2, MDaInF);
			// polar experimental image
			inPolar = imToPolar(MDaExpShiftRot2, first, n_rad);
			applyFourierImage3(inPolar, MDaInAuxF, n_ang);

			// find rotation and shift
			ccMatrix(MDaInAuxF,vecMDaRef_polarF[candidatesFirstLoop[Idx[k]]],ccMatrixRot2);
			maxByColumn(ccMatrixRot2, ccVectorRot2);
			peaksFound = 0;
			std::vector<double> cand(maxAccepted, 0.);
			rotCandidates3(ccVectorRot2, cand, XSIZE(ccMatrixRot2));
			bestCand(MDaExpShiftRot2, MDaInF, vecMDaRef[Idx[k]], cand, psi,Tx, Ty, cc_coeff); //

			// if it's better and x/y shifts are within range then update
			double testShiftTx = bestTx[Idx[k]] + Tx;
			double testShiftTy = bestTy[Idx[k]] + Ty;
			if (cc_coeff > candidatesFirstLoopCoeff[Idx[k]] &&
					std::abs(testShiftTx) < maxShift	&& std::abs(testShiftTy) < maxShift) {
				Idx2[k] = k;
				Idx3[k] = k;
				candidatesSecondLoop[Idx[k]] = candidatesFirstLoop[Idx[k]];
				candidatesSecondLoopCoeff[Idx[k]] = cc_coeff;
				VEC_ELEM(ccvec,Idx[k])=cc_coeff;
				bestTx2[Idx[k]] = testShiftTx;
VEC_ELEM(txvec,Idx[k])=testShiftTx;
				bestTy2[Idx[k]] = testShiftTy;
VEC_ELEM(tyvec,Idx[k])=testShiftTy;
				bestPsi2[Idx[k]] = bestPsi[Idx[k]] + psi;
VEC_ELEM(psivec,Idx[k])=realWRAP(bestPsi[Idx[k]] + psi, -180., 180.); // realWRAP(bestPsi2[Idx2[0]], -180., 180.)
			} else {
				Idx2[k] = k;
				Idx3[k] = k;
				candidatesSecondLoop[Idx[k]] = candidatesFirstLoop[Idx[k]];
				candidatesSecondLoopCoeff[Idx[k]] =candidatesFirstLoopCoeff[Idx[k]];
				VEC_ELEM(ccvec,Idx[k])=candidatesFirstLoopCoeff[Idx[k]];
				bestTx2[Idx[k]] = bestTx[Idx[k]];
VEC_ELEM(txvec,Idx[k])=bestTx[Idx[k]];
				bestTy2[Idx[k]] = bestTy[Idx[k]];
VEC_ELEM(tyvec,Idx[k])=bestTy[Idx[k]];
				bestPsi2[Idx[k]] = bestPsi[Idx[k]];
VEC_ELEM(psivec,Idx[k])=realWRAP(bestPsi[Idx[k]], -180., 180.);
			}
		}// end if(k<nCand)
		else{
			Idx2[k] = k;
			Idx3[k] = k;
			candidatesSecondLoop[Idx[k]] = candidatesFirstLoop[Idx[k]];
			candidatesSecondLoopCoeff[Idx[k]] =candidatesFirstLoopCoeff[Idx[k]];
			VEC_ELEM(ccvec,Idx[k])=candidatesFirstLoopCoeff[Idx[k]];
			bestTx2[Idx[k]] = bestTx[Idx[k]];
VEC_ELEM(txvec,Idx[k])=bestTx[Idx[k]];
			bestTy2[Idx[k]] = bestTy[Idx[k]];
VEC_ELEM(tyvec,Idx[k])=bestTy[Idx[k]];
			bestPsi2[Idx[k]] = bestPsi[Idx[k]];
VEC_ELEM(psivec,Idx[k])=realWRAP(bestPsi[Idx[k]], -180., 180.);
		}
	}

/*comment
ccvec.write("/home/jeison/Escritorio/testVectCC_2ndLoop.txt");
txvec.write("/home/jeison/Escritorio/testVectTx_2ndLoop.txt");
tyvec.write("/home/jeison/Escritorio/testVectTy_2ndLoop.txt");
psivec.write("/home/jeison/Escritorio/testVectPsi_2ndLoop.txt");
// */

	// ================ Graph Filter Process after second loop =================
	Matrix1D<double> ccvec_filt;
	graphFourierFilter(ccvec,ccvec_filt);

Matrix1D<double> txvec_filt;
graphFourierFilter(txvec,txvec_filt);

Matrix1D<double> tyvec_filt;
graphFourierFilter(tyvec,tyvec_filt);

Matrix1D<double> psivec_filt;
graphFourierFilter(psivec,psivec_filt);

/*comment
ccvec_filt.write("/home/jeison/Escritorio/test_ccVect_filt.txt");
txvec_filt.write("/home/jeison/Escritorio/test_txVect_filt.txt");
tyvec_filt.write("/home/jeison/Escritorio/test_tyVect_filt.txt");
psivec_filt.write("/home/jeison/Escritorio/test_psiVect_filt.txt");
// */

	// choose best of the candidates after 2nd loop
	int nCand2 = 1;
	std::partial_sort(Idx2.begin(), Idx2.begin()+nCand2, Idx2.end(),
			[&candidatesSecondLoopCoeff](int i, int j) {return candidatesSecondLoopCoeff[i] > candidatesSecondLoopCoeff[j];});

/*comment
std::cout<<"testIdx2[0]: "<<Idx2[0]<<std::endl;
std::cout<<"testCand[Idx2[0]]: "<<candidatesSecondLoop[Idx2[0]]<<std::endl;
std::cout<<"testCandCoeff[Idx2[0]]: "<<candidatesSecondLoopCoeff[Idx2[0]]<<std::endl;
std::ofstream outCandidateBef("/home/jeison/Escritorio/bestCandidateBefore.txt");
outCandidateBef<<candidatesSecondLoop[Idx2[0]]<<"\n";
outCandidateBef.close();
// */

	// choose best candidate direction from graph filtered ccvect signal
	std::partial_sort(Idx3.begin(), Idx3.begin()+nCand2, Idx3.end(),
			[&ccvec_filt](int i, int j) {return ccvec_filt[i] > ccvec_filt[j];});

/*comment
std::cout<<"testIdx3[0]: "<<Idx3[0]<<std::endl;
std::cout<<"testCand[Idx3[0]]: "<<candidatesSecondLoop[Idx3[0]]<<std::endl;
std::cout<<"testCandCoeff[Idx3[0]]: "<<candidatesSecondLoopCoeff[Idx3[0]]<<std::endl;
std::ofstream outCandidate("/home/jeison/Escritorio/bestCandidateGraph.txt");
outCandidate<<candidatesSecondLoop[Idx3[0]]<<"\n";
outCandidate.close();
// */

	// angular distance between this two direction
	Matrix1D<double> dirj;
	Matrix1D<double> dirjp;
	double rotj;
	double tiltj;
	double psij;
	rotj=referenceRot.at(candidatesSecondLoop[Idx2[0]]);
	tiltj=referenceTilt.at(candidatesSecondLoop[Idx2[0]]);
	psij=0.;
	Euler_direction(rotj, tiltj, psij, dirj);
	double rotjp, tiltjp, psijp;
	rotjp=referenceRot.at(candidatesSecondLoop[Idx3[0]]);
	tiltjp=referenceTilt.at(candidatesSecondLoop[Idx3[0]]);
	psijp=0.;
	Euler_direction(rotjp, tiltjp, psijp, dirjp);
	double sphericalDistance=RAD2DEG(spherical_distance(dirj, dirjp));

//double energyConcentration2ndLoop=energyDistribution(dirj,Idx2,candidatesSecondLoopCoeff,candidatesSecondLoop);

//double energyConcentrationFiltered=energyDistribution(dirj,Idx3,ccvec_filt);


/*// BAD RESULTS replacing alignment parameters after 2nd loop by filtered values
if (sphericalDistance<maxDistance){ //replace with alignment parameters from filtered versions
bestTx2[Idx2[0]]=VEC_ELEM(txvec_filt,Idx2[0]);
bestTy2[Idx2[0]]=VEC_ELEM(tyvec_filt,Idx2[0]);
bestPsi2[Idx2[0]]=VEC_ELEM(psivec_filt,Idx2[0]);
}
else{
candidatesSecondLoopCoeff[Idx2[0]]=exp(-.5*sphericalDistance/maxDistance);
}// */

	// is this direction a reliable candidate?
	// a condition could be: if this direction is within "soft neighborhood" with high coeff values
	// BETTER RESULTS
	double maxDistance=3.*angStep;
	if(sphericalDistance>maxDistance)
		candidatesSecondLoopCoeff[Idx2[0]]=exp(-.5*sphericalDistance/maxDistance);



/* time
double duration = ( std::clock() - Inicio ) / (double) CLOCKS_PER_SEC;
std::cout << "Operation in processImage took "<< duration*1000 << "milliseconds" << std::endl; // */

/*comment
std::cout<<"sphericalDistance: "<<sphericalDistance<<std::endl;
std::cout<<"energyConcentration2ndLoop: "<<energyConcentration2ndLoop/sizeMdRef<<std::endl;
std::cout<<"energyConcentrationFiltered: "<<energyConcentrationFiltered/sizeMdRef<<std::endl;
// */

//std::ofstream outTx;
//std::ofstream outTy;
//std::ofstream outPsi;
//outTx.open("/home/jeison/Escritorio/outTx.txt", std::ios::app);
//outTy.open("/home/jeison/Escritorio/outTy.txt", std::ios::app);
//outPsi.open("/home/jeison/Escritorio/outPsi.txt", std::ios::app);
//outTx<<sphericalDistance<<"\t"<<std::abs(bestTx2[Idx2[0]]-VEC_ELEM(txvec_filt,Idx2[0]))<<"\t"<<bestTx2[Idx2[0]]<<"\t"<<VEC_ELEM(txvec_filt,Idx2[0]) << "\n";
//outTy<<sphericalDistance<<"\t"<<std::abs(bestTy2[Idx2[0]]-VEC_ELEM(tyvec_filt,Idx2[0]))<<"\t"<<bestTy2[Idx2[0]]<<"\t"<<VEC_ELEM(tyvec_filt,Idx2[0]) << "\n";
//outPsi<<sphericalDistance<<"\t"<<std::abs(bestPsi2[Idx2[0]]-VEC_ELEM(psivec_filt,Idx2[0]))<<"\t"<<bestPsi2[Idx2[0]]<<"\t"<<VEC_ELEM(psivec_filt,Idx2[0]) << "\n";
//outTx.close();
//outTy.close();
//outPsi.close();
//
//std::cout<<Idx2[0]<<"-->"<<candidatesSecondLoop[Idx2[0]]<<"\n";
//std::cout<<"sphericalDistance: "<<sphericalDistance<<"\n";

/*exit
if(testCounter2==0)
	exit(1);
testCounter2+=1;// */

	// reading info of reference image candidate
	double rotRef;
	double tiltRef;
	// reading info of reference image candidate
	rotRef = referenceRot.at(candidatesSecondLoop[Idx2[0]]);
	tiltRef = referenceTilt.at(candidatesSecondLoop[Idx2[0]]);
	//save metadata of images with angles
	rowOut.setValue(MDL_IMAGE, fnImgOut);
	rowOut.setValue(MDL_ENABLED, 1);
	rowOut.setValue(MDL_MAXCC, candidatesSecondLoopCoeff[Idx2[0]]);
	rowOut.setValue(MDL_WEIGHT, 1.);
	rowOut.setValue(MDL_WEIGHT_SIGNIFICANT, 1.);
	rowOut.setValue(MDL_ANGLE_ROT, rotRef);
	rowOut.setValue(MDL_ANGLE_TILT, tiltRef);
	rowOut.setValue(MDL_ANGLE_PSI, realWRAP(bestPsi2[Idx2[0]], -180., 180.));
	rowOut.setValue(MDL_SHIFT_X, -bestTx2[Idx2[0]]);
	rowOut.setValue(MDL_SHIFT_Y, -bestTy2[Idx2[0]]);


}

void ProgAngularAssignmentMag::postProcess() {
	// from angularContinousAssign2
	MetaData &ptrMdOut = *getOutputMd();
	ptrMdOut.removeDisabled();
	double maxCC = -1.;
	 FOR_ALL_OBJECTS_IN_METADATA(ptrMdOut){
		 double thisMaxCC;
		 ptrMdOut.getValue(MDL_MAXCC, thisMaxCC, __iter.objId);
		 if (thisMaxCC > maxCC)
			 maxCC = thisMaxCC;
		 if (thisMaxCC == 0)
			 ptrMdOut.removeObject(__iter.objId);
	 }
	 FOR_ALL_OBJECTS_IN_METADATA(ptrMdOut){
		 double thisMaxCC;
		 ptrMdOut.getValue(MDL_MAXCC, thisMaxCC, __iter.objId);
		 ptrMdOut.setValue(MDL_WEIGHT, thisMaxCC / maxCC, __iter.objId);
		 ptrMdOut.setValue(MDL_WEIGHT_SIGNIFICANT, thisMaxCC / maxCC, __iter.objId);
	 }

	ptrMdOut.write(XmippMetadataProgram::fn_out.replaceExtension("xmd"));
}

/* Pearson Coeff. ZNCC zero-mean normalized cross-corr*/
void ProgAngularAssignmentMag::pearsonCorr(const MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff){
	// covariance
	double X_m;
	double Y_m;
	double X_std;
	double Y_std;
	arithmetic_mean_and_stddev(X, X_m, X_std);
	arithmetic_mean_and_stddev(Y, Y_m, Y_std);

	double mean_prod = mean_of_products(X, Y);
	double covariace = mean_prod - (X_m * Y_m);

	coeff = covariace / (X_std * Y_std);
}

/* Arithmetic mean and stdDev for Pearson Coeff */
void ProgAngularAssignmentMag::arithmetic_mean_and_stddev(
		const MultidimArray<double> &data, double &avg, double &stddev) {
	data.computeAvgStdev(avg, stddev);
}

/* Arithmetic mean and stdDev for Pearson Coeff */
void ProgAngularAssignmentMag::arithmetic_mean_and_stddev(
		MultidimArray<double> &data, double &avg, double &stddev) {
	data.computeAvgStdev(avg, stddev);
}

/* Mean of products for Pearson Coeff */
double ProgAngularAssignmentMag::mean_of_products(
		const MultidimArray<double> &data1, MultidimArray<double> &data2) {
	double total = 0;
	for (int f = 0; f < Ydim; f++) {
		for (int c = 0; c < Xdim; c++) {
			total += DIRECT_A2D_ELEM(data1,f,c) * DIRECT_A2D_ELEM(data2, f, c);
		}
	}
	return total / (Xdim * Ydim);
}

/* Mean of products for Pearson Coeff */
double ProgAngularAssignmentMag::mean_of_products(MultidimArray<double> &data1,
		MultidimArray<double> &data2) {
	double total = 0;
	for (int f = 0; f < Ydim; f++) {
		for (int c = 0; c < Xdim; c++) {
			total += DIRECT_A2D_ELEM(data1,f,c) * DIRECT_A2D_ELEM(data2, f, c);
		}
	}
	return total / (Xdim * Ydim);
}

/* Normalized cross correlation*/
void ::ProgAngularAssignmentMag::normalized_cc(MultidimArray<double> &X,
		MultidimArray<double> &Y, double &value) {
	double prodXY = 0;
	double prodXX = 0;
	double prodYY = 0;
	MultidimArray<double> X2;
	MultidimArray<double> Y2;
	X2.resizeNoCopy(X);
	Y2.resizeNoCopy(Y);

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(X){
		prodXY += dAij(X,i,j) * dAij(Y, i, j);
		prodXX += dAij(X,i,j) * dAij(X, i, j);
		prodYY += dAij(Y,i,j) * dAij(Y, i, j);
	}
	double den = prodXX * prodYY;
	if (den <= 0)
		std::cout << "zero/negative denominator!!\n" << std::endl;
	else
		value = prodXY / sqrt(den);
}

/* Normalized cross correlation*/
void ::ProgAngularAssignmentMag::normalized_cc(const MultidimArray<double> &X,
		MultidimArray<double> &Y, double &value) {
	double prodXY = 0;
	double prodXX = 0;
	double prodYY = 0;
	MultidimArray<double> X2;
	MultidimArray<double> Y2;
	X2.resizeNoCopy(X);
	Y2.resizeNoCopy(Y);

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(X){
		prodXY += dAij(X,i,j) * dAij(Y, i, j);
		prodXX += dAij(X,i,j) * dAij(X, i, j);
		prodYY += dAij(Y,i,j) * dAij(Y, i, j);
	}
	double den = prodXX * prodYY;
	if (den <= 0)
		std::cout << "zero/negative denominator!!\n" << std::endl;
	else
		value = prodXY / sqrt(den);
}

/* mixed between IMED and NCC --> IMNCC ----------------------------------------------- */
void ProgAngularAssignmentMag::imNormalized_cc(const MultidimArray<double>& I1,
		const MultidimArray<double>& I2, double &value) {
	// [x,y]=meshgrid([-3:1:3],[-3:1:3])
	// format long	// w=1/sqrt(2*pi)*exp(-0.5*(x.*x+y.*y))
	double *refW;
	double w[49] = { 0.000049233388666, 0.000599785460091, 0.002688051941039,
			0.004431848411938, 0.002688051941039, 0.000599785460091,
			0.000049233388666, 0.000599785460091, 0.007306882745281,
			0.032747176537767, 0.053990966513188, 0.032747176537767,
			0.007306882745281, 0.000599785460091, 0.002688051941039,
			0.032747176537767, 0.146762663173740, 0.241970724519143,
			0.146762663173740, 0.032747176537767, 0.002688051941039,
			0.004431848411938, 0.053990966513188, 0.241970724519143,
			0.398942280401433, 0.241970724519143, 0.053990966513188,
			0.004431848411938, 0.002688051941039, 0.032747176537767,
			0.146762663173740, 0.241970724519143, 0.146762663173740,
			0.032747176537767, 0.002688051941039, 0.000599785460091,
			0.007306882745281, 0.032747176537767, 0.053990966513188,
			0.032747176537767, 0.007306882745281, 0.000599785460091,
			0.000049233388666, 0.000599785460091, 0.002688051941039,
			0.004431848411938, 0.002688051941039, 0.000599785460091,
			0.000049233388666 };

	int imiddle = YSIZE(I1) / 2;
	int jmiddle = XSIZE(I1) / 2;
	int R2max = imiddle * imiddle;
	int ysize = (int) YSIZE(I1);
	int xsize = (int) XSIZE(I1);

	double numXY = 0.;
	double denXX = 0.;
	double denYY = 0.;

	MultidimArray<double> prodImageXY = I1 * I2;
	MultidimArray<double> prodImageXX = I1 * I1;
	MultidimArray<double> prodImageYY = I2 * I2;

	for (int i = 3; i < ysize - 3; ++i) {
		int i2 = (i - imiddle) * (i - imiddle);
		for (int j = 3; j < xsize - 3; ++j) {
			int j2 = (j - jmiddle) * (j - jmiddle);
			if (i2 + j2 > R2max) // Measure only within the maximum circle
				continue;

			double prodNumXYi = DIRECT_A2D_ELEM(prodImageXY, i, j);
			double prodDenXXi = DIRECT_A2D_ELEM(prodImageXX, i, j);
			double prodDenYYi = DIRECT_A2D_ELEM(prodImageYY, i, j);
			int index = 0;
			for (int ii = -3; ii <= 3; ++ii) {
				refW = &w[index];
				index = index + 7;
				// numerator
				double *prodNumXYj = &DIRECT_A2D_ELEM(prodImageXY, i + ii,
						j - 3);
				double prodNumXYAux = (*refW) * (*prodNumXYj++);
				// Denominator XX
				double *prodDenXXj = &DIRECT_A2D_ELEM(prodImageXX, i + ii,
						j - 3);
				double prodDenXXAux = (*refW) * (*prodDenXXj++);
				// Denominator YY
				double *prodDenYYj = &DIRECT_A2D_ELEM(prodImageYY, i + ii,
						j - 3);
				double prodDenYYAux = (*refW++) * (*prodDenYYj++); //increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				numXY += prodNumXYAux * prodNumXYi;
				denXX += prodDenXXAux * prodDenXXi;
				denYY += prodDenYYAux * prodDenYYi;
			}
		}
	}

	double denom = denXX * denYY;
	value = 0;
	if (denom <= 0)
		std::cout << "zero/negative denominator!" << std::endl;
	else
		value = numXY / sqrt(denom);

}

/* mixed between IMED and ZNCC --> IMNCC ----------------------------------------------- */
void ProgAngularAssignmentMag::imZNCC(const MultidimArray<double>& I1,
		const MultidimArray<double>& I2, double &value) {

	// [x,y]=meshgrid([-3:1:3],[-3:1:3])
	// format long	// w=1/sqrt(2*pi)*exp(-0.5*(x.*x+y.*y))
	double *refW;
	double w[49] = { 0.000049233388666, 0.000599785460091, 0.002688051941039,
			0.004431848411938, 0.002688051941039, 0.000599785460091,
			0.000049233388666, 0.000599785460091, 0.007306882745281,
			0.032747176537767, 0.053990966513188, 0.032747176537767,
			0.007306882745281, 0.000599785460091, 0.002688051941039,
			0.032747176537767, 0.146762663173740, 0.241970724519143,
			0.146762663173740, 0.032747176537767, 0.002688051941039,
			0.004431848411938, 0.053990966513188, 0.241970724519143,
			0.398942280401433, 0.241970724519143, 0.053990966513188,
			0.004431848411938, 0.002688051941039, 0.032747176537767,
			0.146762663173740, 0.241970724519143, 0.146762663173740,
			0.032747176537767, 0.002688051941039, 0.000599785460091,
			0.007306882745281, 0.032747176537767, 0.053990966513188,
			0.032747176537767, 0.007306882745281, 0.000599785460091,
			0.000049233388666, 0.000599785460091, 0.002688051941039,
			0.004431848411938, 0.002688051941039, 0.000599785460091,
			0.000049233388666 };

	int imiddle = YSIZE(I1) / 2;
	int jmiddle = XSIZE(I1) / 2;
	int R2max = imiddle * imiddle;
	int ysize = (int) YSIZE(I1);
	int xsize = (int) XSIZE(I1);

	double numXY = 0.;
	double denXX = 0.;
	double denYY = 0.;

	//compute mean
	double avgI1 = I1.computeAvg();
	double avgI2 = I2.computeAvg();

	MultidimArray<double> centeredI1 = I1 - avgI1;
	MultidimArray<double> centeredI2 = I2 - avgI2;

	MultidimArray<double> prodImageXY = centeredI1 * centeredI2;
	MultidimArray<double> prodImageXX = centeredI1 * centeredI1;
	MultidimArray<double> prodImageYY = centeredI2 * centeredI2;

	for (int i = 3; i < ysize - 3; ++i) {
		int i2 = (i - imiddle) * (i - imiddle);
		for (int j = 3; j < xsize - 3; ++j) {
			int j2 = (j - jmiddle) * (j - jmiddle);
			if (i2 + j2 > R2max) // Measure only within the maximum circle
				continue;
			//        	 /* using one loop
			double prodNumXYi = DIRECT_A2D_ELEM(prodImageXY, i, j);
			double prodDenXXi = DIRECT_A2D_ELEM(prodImageXX, i, j);
			double prodDenYYi = DIRECT_A2D_ELEM(prodImageYY, i, j);
			int index = 0;
			for (int ii = -3; ii <= 3; ++ii) {
				refW = &w[index];
				index = index + 7;
				// numerator
				double *prodNumXYj = &DIRECT_A2D_ELEM(prodImageXY, i + ii,
						j - 3);
				double prodNumXYAux = (*refW) * (*prodNumXYj++);
				// Denominator XX
				double *prodDenXXj = &DIRECT_A2D_ELEM(prodImageXX, i + ii,
						j - 3);
				double prodDenXXAux = (*refW) * (*prodDenXXj++);
				// Denominator YY
				double *prodDenYYj = &DIRECT_A2D_ELEM(prodImageYY, i + ii,
						j - 3);
				double prodDenYYAux = (*refW++) * (*prodDenYYj++); //increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				prodNumXYAux += (*refW) * (*prodNumXYj++);
				prodDenXXAux += (*refW) * (*prodDenXXj++);
				prodDenYYAux += (*refW++) * (*prodDenYYj++); // increment

				numXY += prodNumXYAux * prodNumXYi;
				denXX += prodDenXXAux * prodDenXXi;
				denYY += prodDenYYAux * prodDenYYi;
			} // */
		}
	}

	double denom = denXX * denYY;
	value = 0;
	if (denom <= 0)
		std::cout << "zero/negative denominator!" << std::endl;
	else
		value = numXY / sqrt(denom);
}

void ProgAngularAssignmentMag::applyCircularMask(
		const MultidimArray<double> &in, MultidimArray<double> &out) {

	double Cf = (Ydim + (Ydim % 2)) / 2.0; // for odd/even images
	double Cc = (Xdim + (Xdim % 2)) / 2.0;
	int pixReduc = 2;
	double rad2 = (Cf - pixReduc) * (Cf - pixReduc);
	double val = 0;
	out.initZeros(Ydim, Xdim);
	for (size_t f = 0; f < Ydim; f++) {
		for (size_t c = 0; c < Xdim; c++) {
			val = (f - Cf) * (f - Cf) + (c - Cc) * (c - Cc);
			if (val < rad2)
				DIRECT_A2D_ELEM(out, f, c) = DIRECT_A2D_ELEM(in, f, c);
		}
	}
}

/* writing out some data to file with an specified size*/
void ProgAngularAssignmentMag::_writeTestFile(MultidimArray<double> &data,
		const char* fileName, size_t nFil, size_t nCol) {
	std::ofstream outFile(fileName);
	for (int f = 0; f < nFil; f++) {
		for (int c = 0; c < nCol; c++) {
			outFile << DIRECT_A2D_ELEM(data, f, c) << "\t";
		}
		outFile << "\n";
	}
	outFile.close();
}

/* writing out some data to file with an specified size*/
void ProgAngularAssignmentMag::_writeTestFile(const MultidimArray<double> &data,
		const char* fileName, size_t nFil, size_t nCol) {
	std::ofstream outFile(fileName);
	for (int f = 0; f < nFil; f++) {
		for (int c = 0; c < nCol; c++) {
			outFile << DIRECT_A2D_ELEM(data, f, c) << "\t";
		}
		outFile << "\n";
	}
	outFile.close();
}

/* writing out some data to file Ydim x Xdim size*/
void ProgAngularAssignmentMag::_writeTestFile(MultidimArray<double> &data,
		const char* fileName) {
	std::ofstream outFile(fileName);
	for (int f = 0; f < Ydim; f++) {
		for (int c = 0; c < Xdim; c++) {
			outFile << DIRECT_A2D_ELEM(data, f, c) << "\t";
		}
		outFile << "\n";
	}
	outFile.close();
}

/* get COMPLETE fourier spectrum of Images. It should be changed for half */
void ProgAngularAssignmentMag::applyFourierImage(MultidimArray<double> &data,
		MultidimArray<std::complex<double> > &FourierData) {
	transformerImage.completeFourierTransform(data, FourierData);
}

/* get COMPLETE fourier spectrum of polarRepresentation of Magnitude. It should be changed for half */
void ProgAngularAssignmentMag::applyFourierImage(MultidimArray<double> &data,
		MultidimArray<std::complex<double> > &FourierData, const size_t &ang) {
	(void)ang;
	transformerPolarImage.completeFourierTransform(data, FourierData);
}

/*first try in using only one half of Fourier space*/
void ProgAngularAssignmentMag::applyFourierImage2(MultidimArray<double> &data,
		MultidimArray<std::complex<double> > &FourierData) {
	transformerImage.FourierTransform(data, FourierData, true);
}

/* first try one half of fourier spectrum of polarRepresentation of Magnitude*/
void ProgAngularAssignmentMag::applyFourierImage2(MultidimArray<double> &data,
		MultidimArray<std::complex<double> > &FourierData, const size_t &ang) {
	(void)ang;
	transformerPolarImage.FourierTransform(data, FourierData, true); // false --> true para generar copia
}

/* first try one half of fourier spectrum of polarRepresentation of image in real space*/
void ProgAngularAssignmentMag::applyFourierImage3(MultidimArray<double> &data,
		MultidimArray<std::complex<double> > &FourierData, const size_t &ang) {
	(void)ang;
	transformerPolarRealSpace.FourierTransform(data, FourierData, true); //
}

/* get magnitude of fourier spectrum */
void ProgAngularAssignmentMag::getComplexMagnitude(
		MultidimArray<std::complex<double> > &FourierData,
		MultidimArray<double> &FourierMag) {
	FFT_magnitude(FourierData, FourierMag);
}

/* cartImg contains cartessian  grid representation of image,
 *  rad and ang are the number of radius and angular elements*/
MultidimArray<double> ProgAngularAssignmentMag::imToPolar(MultidimArray<double> &cartIm, size_t &start, size_t &end) {
	int thisNbands = end - start;
	MultidimArray<double> polarImg(thisNbands, n_ang2);
	float pi = 3.141592653;
	// coordinates of center
	double cy = (Ydim + (Ydim % 2)) / 2.0; // for odd/even images
	double cx = (Xdim + (Xdim % 2)) / 2.0;

	// scale factors
	double sfy = (Ydim - 1) / 2.0;
	double sfx = (Xdim - 1) / 2.0;

	double delR = (double) (1.0 / (n_rad)); // n_rad-1
	double delT = 2.0 * pi / n_ang2;

	// loop through rad and ang coordinates
	double r;
	double t;
	double x_coord;
	double y_coord;
	for (size_t ri = start; ri < end; ++ri) {
		for (size_t ti = 0; ti < n_ang2; ++ti) {
			r = ri * delR;
			t = ti * delT;
			x_coord = (r * cos(t)) * sfx + cx;
			y_coord = (r * sin(t)) * sfy + cy;
			// set value of polar img
			DIRECT_A2D_ELEM(polarImg,ri-start,ti) = interpolate(cartIm,x_coord,y_coord);
		}
	}

	return polarImg;
}

/* cartImg contains cartessian  grid representation of image,
 *  rad and ang are the number of radius and angular elements
 *  this function was built for half representation of Fourier spectrum*/
MultidimArray<double> ProgAngularAssignmentMag::imToPolar2(
		MultidimArray<double> &cartIm, const size_t &rad, const size_t &ang) {
	MultidimArray<double> polarImg(rad, ang);
	float pi = 3.141592653;
	// coordinates of center
	double cy = 0.5;
	double cx = (Xdim + 1) / 2.0;
	// scale factors
	double sfy = (Ydim - 1) / 2.0;
	double sfx = (Xdim - 1) / 2.0;

	double delR = (double) (1.0 / (rad - 1));
	double delT = pi / ang;

	// loop through rad and ang coordinates
	double r;
	double t;
	double x_coord;
	double y_coord;
	for (size_t ri = 0; ri < rad; ++ri) {
		for (size_t ti = 0; ti < ang; ++ti) {
			r = ri * delR;
			t = ti * delT;
			x_coord = (r * cos(t)) * sfx + cx;
			y_coord = (r * sin(t)) * sfy + cy;

			// set value of polar img
			DIRECT_A2D_ELEM(polarImg,ri,ti) = interpolate(cartIm,x_coord,y_coord);
		}
	}

	return polarImg;
}

/* bilinear interpolation */
double ProgAngularAssignmentMag::interpolate(MultidimArray<double> &cartIm,
		double &x_coord, double &y_coord) {
	double val;
	size_t xf = floor(x_coord);
	size_t xc = ceil(x_coord);
	size_t yf = floor(y_coord);
	size_t yc = ceil(y_coord);

	if ((xf == xc) && (yf == yc)) {
		val = dAij(cartIm, xc, yc);
	} else if (xf == xc) { // linear
		val = dAij(cartIm, xf, yf)
				+ (y_coord - yf)
						* ( dAij(cartIm, xf, yc) - dAij(cartIm, xf, yf));
	} else if (yf == yc) { // linear
		val = dAij(cartIm, xf, yf)
				+ (x_coord - xf)
						* ( dAij(cartIm, xc, yf) - dAij(cartIm, xf, yf));
	} else { // bilinear
		val = ((double) (( dAij(cartIm,xf,yf) * (yc - y_coord) + dAij(cartIm,xf,yc) * (y_coord - yf)) * (xc - x_coord))
						+ (double) (( dAij(cartIm,xc,yf) * (yc - y_coord)+ dAij(cartIm,xc,yc) * (y_coord - yf))	* (x_coord - xf)))
						/ (double) ((xc - xf) * (yc - yf));
	}

	return val;
}

/* implementing centering using circular-shift*/
void ProgAngularAssignmentMag::completeFourierShift(MultidimArray<double> &in,
		MultidimArray<double> &out) {

	// correct output size
	out.resizeNoCopy(in);

	size_t Cf = (size_t) (YSIZE(in) / 2.0 + 0.5); //(Ydim/2.0 + 0.5); //FIXME I think this does not work properly for even/odd images
	size_t Cc = (size_t) (XSIZE(in) / 2.0 + 0.5); //(Xdim/2.0 + 0.5); // (size_t)( 120 / 2.0 + 0.5) == (size_t)( 119 / 2.0 + 0.5)

//	size_t Cf = (size_t) ((YSIZE(in) + (YSIZE(in) % 2)) / 2.0); // it seems not different
//	size_t Cc = (size_t) ((XSIZE(in) + (XSIZE(in) % 2)) / 2.0);

	size_t ff;
	size_t cc;
	for (size_t f = 0; f < YSIZE(in); f++) {
		ff = (f + Cf) % YSIZE(in);
		for (size_t c = 0; c < XSIZE(in); c++) {
			cc = (c + Cc) % XSIZE(in);
			DIRECT_A2D_ELEM(out, ff, cc) = DIRECT_A2D_ELEM(in, f, c);
		}
	}
}

/* its an experiment for implement fftshift*/
void ProgAngularAssignmentMag::halfFourierShift(MultidimArray<double> &in,
		MultidimArray<double> &out) {
	size_t Cf = (size_t) (Ydim / 2.0 + 0.5);
	out.resizeNoCopy(in);

	size_t ff;
	size_t cc;
	for (size_t f = 0; f < Ydim; f++) {
		ff = (f + Cf) % Ydim;
		for (size_t c = 0; c < Cf; c++) {
			cc = c;
			DIRECT_A2D_ELEM(out, ff, cc) = DIRECT_A2D_ELEM(in, f, c);
		}
	}

}

/*
 * experiment for GCC matrix product F1 .* conj(F2) using Compactly supported approach
 */
void ProgAngularAssignmentMag::ccMatrix(const MultidimArray<std::complex<double>> &F1,
		const MultidimArray<std::complex<double>> &F2,/*reference image*/
		MultidimArray<double> &result) {

	result.resizeNoCopy(YSIZE(F1), 2 * (XSIZE(F1) - 1));
	//result.resizeNoCopy(vecMDaRef[0]); // NO!

/*comment
if(testCounter==2){
std::cout<<"size in ccMatrix of F1: "<< YSIZE(F1) <<" x "<< XSIZE(F1) << std::endl;
std::cout<<"size in ccMatrix full: "<< YSIZE(F1) <<" x "<< 2 * (XSIZE(F1) - 1) << std::endl;
} // */

	CorrelationAux aux;
	aux.transformer1.setReal(result);
	aux.transformer1.setFourier(F1);
	// Multiply FFT1 .* FFT2'
	double a;
	double b;
	double c;
	double d; // a+bi, c+di
	double dSize = MULTIDIM_SIZE(result);

	double *ptrFFT2 = (double*) MULTIDIM_ARRAY(F2);
	double *ptrFFT1 = (double*) MULTIDIM_ARRAY(aux.transformer1.fFourier);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F1)
	{
		a = (*ptrFFT1) * dSize;
		b = (*(ptrFFT1 + 1)) * dSize;
		c = (*ptrFFT2++);
		d = (*ptrFFT2++) * (-1);
		//Compactly supported correlation. F2 is reference image
		*ptrFFT1++ = (a * c - b * d) / ((c * c + d * d) + 0.001);
		*ptrFFT1++ = (b * c + a * d) / ((c * c + d * d) + 0.001);
	}
	aux.transformer1.inverseFourierTransform();
	CenterFFT(result, true);
	result.setXmippOrigin();
}

/*
 * PhaseCorr only for shift  ( F1 .* conj(F2) ) ./ ||  F1 .* conj(F2) ||²
 */
void ProgAngularAssignmentMag::ccMatrixPCO(const MultidimArray<std::complex<double>> &F1,
		const MultidimArray<std::complex<double>> &F2,
		MultidimArray<double> &result) {

	result.resizeNoCopy(YSIZE(F1), 2 * (XSIZE(F1) - 1));

	CorrelationAux aux;
	aux.transformer1.setReal(result);
	aux.transformer1.setFourier(F1);
	// Multiply FFT1 * FFT2'
	double a;
	double b;
	double c;
	double d; // a+bi, c+di
	double *ptrFFT2 = (double*) MULTIDIM_ARRAY(F2);
	double *ptrFFT1 = (double*) MULTIDIM_ARRAY(aux.transformer1.fFourier);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F1)
	{
		a = *ptrFFT1;
		b = *(ptrFFT1 + 1);
		c = (*ptrFFT2++);
		d = (*ptrFFT2++) * (-1);
		// phase corr only
		double den = (a * c - b * d) * (a * c - b * d) + (b * c + a * d) * (b * c + a * d);
		*ptrFFT1++ = (a * c - b * d) / (den + 0.001);
		*ptrFFT1++ = (b * c + a * d) / (den + 0.001);
	}

	aux.transformer1.inverseFourierTransform();
	CenterFFT(result, true);
	result.setXmippOrigin();
}


/* gets maximum value for each column*/
void ProgAngularAssignmentMag::maxByColumn(MultidimArray<double> &in,
		MultidimArray<double> &out) {

	out.resizeNoCopy(1, XSIZE(in));
	double maxVal;
	double val2;
	for (int c = 0; c < XSIZE(in); c++) {
		maxVal = dAij(in, 0, c);
		for (int f = 1; f < YSIZE(in); f++) {
			val2 = dAij(in, f, c);
			if (val2 > maxVal)
				maxVal = val2;
		}
		dAi(out,c) = maxVal;
	}
}

/* gets maximum value for each column*/
void ProgAngularAssignmentMag::meanByColumn(MultidimArray<double> &in,
		MultidimArray<double> &out) {

	out.resizeNoCopy(1, XSIZE(in));
	double val;
	double val2;
	int factor = YSIZE(in);
	for (int c = 0; c < XSIZE(in); c++) {
		val = dAij(in, 0, c);
		for (int f = 1; f < YSIZE(in); f++) {
			val2 = dAij(in, f, c);
			val += val2 / factor;
		}
		dAi(out,c) = val;
	}
}

/* gets maximum value for each row */
void ProgAngularAssignmentMag::maxByRow(MultidimArray<double> &in,
		MultidimArray<double> &out) {
	out.resizeNoCopy(1, YSIZE(in));
	double maxVal;
	double val2;
	for (int f = 0; f < YSIZE(in); f++) {
		maxVal = dAij(in, f, 0);
		for (int c = 1; c < XSIZE(in); c++) {
			val2 = dAij(in, f, c);
			if (val2 > maxVal)
				maxVal = val2;
		}
		dAi(out,f) = maxVal;
	}
}

/* gets maximum value for each row */
void ProgAngularAssignmentMag::meanByRow(MultidimArray<double> &in,
		MultidimArray<double> &out) {
	out.resizeNoCopy(1, YSIZE(in));
	double val;
	double val2;
	int factor = XSIZE(in);
	for (int f = 0; f < YSIZE(in); f++) {
		val = dAij(in, f, 0);
		for (int c = 1; c < XSIZE(in); c++) {
			val2 = dAij(in, f, c);
			val += val2 / factor;
		}
		dAi(out,f) = val;
	}
}

/*quadratic interpolation for location of peak in crossCorr vector*/
double quadInterp(/*const*/int idx, MultidimArray<double> &in) {
	int idxBef = idx - 1;
	if (idxBef < 0)
		std::cout << "Heeey!! ... look at quadInterp method\n";
	double InterpIdx = idx	- (( dAi(in,idx+1) - dAi(in, idx - 1))/ ( dAi(in,idx+1) + dAi(in, idx - 1) - 2 * dAi(in, idx)))/ 2.;
	return InterpIdx;
}

/* precompute Hann 2D window Ydim x Xdim*/
void ProgAngularAssignmentMag::computeHann() {
	float pi = 3.141592653;
	W.resizeNoCopy(Ydim, Xdim);
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(W)
	{
		dAij(W,i,j) = 0.25 * (1 - cos(2 * pi * i / Xdim))
				* (1 - cos(2 * pi * j / Ydim));
	}
}

/*apply hann window to input image*/
void ProgAngularAssignmentMag::hannWindow(MultidimArray<double> &in) {
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(in)
	{
		dAij(in,i,j) *= dAij(W, i, j);
	}
}

/* precompute circular 2D window Ydim x Xdim*/
void ProgAngularAssignmentMag::computeCircular() {

	double Cf = (Ydim + (Ydim % 2)) / 2.0; // for odd/even images
	double Cc = (Xdim + (Xdim % 2)) / 2.0;
	int pixReduc = 1;
	double rad2 = (Cf - pixReduc) * (Cf - pixReduc);
	double val = 0;

	C.resizeNoCopy(Ydim, Xdim);
	C.initZeros(Ydim, Xdim);
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(C)
	{
		val = (j - Cf) * (j - Cf) + (i - Cc) * (i - Cc);
		if (val < rad2)
			dAij(C,i,j) = 1.;
	}
}

/*apply circular window to input image*/
void ProgAngularAssignmentMag::circularWindow(MultidimArray<double> &in) {
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(in)
	{
		dAij(in,i,j) *= dAij(C, i, j);
	}
}

/* Only for 180 angles */
/* approach which selects only two locations of maximum peaks in ccvRot */
void ProgAngularAssignmentMag::rotCandidates3(MultidimArray<double> &in,
		std::vector<double> &cand, const size_t &size) {
	double max1 = -1000.;
	int idx1 = 0;
	double max2 = -1000.;
	int idx2 = 0;
	int cont = 0;
	peaksFound = cont;

	for (int i = 89; i < 272; ++i) { // only look for in range 90:-90
		// current value is a peak value?
		if ((dAi(in,size_t(i)) > dAi(in, size_t(i - 1))) &&
				(dAi(in,size_t(i)) > dAi(in, size_t(i + 1)))) {
			if ( dAi(in,i) > max1) {
				max2 = max1;
				idx2 = idx1;
				max1 = dAi(in, i);
				idx1 = i;
				cont += 1;
			} else if ( dAi(in,i) > max2 && dAi(in,i) != max1) {
				max2 = dAi(in, i);
				idx2 = i;
				cont += 1;
			}
		}
	}
	if (idx1 != 0) {
		int maxAccepted = 1;
		std::vector<int> temp;
		if (idx2 != 0) {
			maxAccepted = 2;
			temp.resize(maxAccepted);
			temp[0] = idx1;
			temp[1] = idx2;
		} else {
			temp.resize(maxAccepted);
			temp[0] = idx1;
		}

		int tam = 2 * maxAccepted;
		peaksFound = tam;
		double interpIdx; // quadratic interpolated location of peak
		for (int i = 0; i < maxAccepted; ++i) {
			interpIdx = quadInterp(temp[i], in);
			cand[i] = double(size) / 2. - interpIdx;
			cand[i + maxAccepted] =	(cand[i] >= 0) ? cand[i] + 180. : cand[i] - 180.;
		}
	} else {
		peaksFound = 0;
	}
}

/* several candidates to best angle psi between images
 * then they are sorted and only keep maxAccepted
 * */
void ProgAngularAssignmentMag::rotCandidates(MultidimArray<double> &in,
		std::vector<double> &cand, const size_t &size) {
	int maxAccepted = 4;
	int maxNumOfPeaks = 90;
	std::vector<int> peakPos(maxNumOfPeaks, 0);
	std::vector<int> peakIdx(maxNumOfPeaks, 0);
	int cont = 0;
	peaksFound = cont;
	for (int i = 89/*1*/; i < 272/*size-1*/; ++i) { // check only the range -90:90
		if ((dAi(in,i) > dAi(in, i - 1)) && (dAi(in,i) > dAi(in, i + 1))) {
			peakIdx[cont] = cont; // for posterior ordering
			peakPos[cont] = i; // position of peak
			cont++;
			peaksFound = cont;
		}
	}

	maxAccepted = (peaksFound < maxAccepted) ? peaksFound : maxAccepted;

	if (cont) {
		//change for partial sort
		std::partial_sort(peakIdx.begin(), peakIdx.begin()+maxAccepted, peakIdx.end(),
				[&in](int i, int j){return dAi(in,i) > dAi(in,j); }); //

		int tam = 2 * maxAccepted; //
		peaksFound = tam;
		double interpIdx;
		for (int i = 0; i < maxAccepted; ++i) {
			interpIdx = quadInterp(peakPos[peakIdx[i]], in);
			cand[i] = double(size) / 2. - interpIdx;
			cand[i + maxAccepted] =	(cand[i] >= 0) ? cand[i] + 180. : cand[i] - 180.;
		}
	} else {
		peaksFound = 0;
	}
}


/* selection of best candidate to rotation and its corresponding shift
 * called at first loop in "coarse" searching
 * shifts are computed as maximum of CrossCorr vector
 * vector<double> cand contains candidates to relative rotation between images
 */
void ProgAngularAssignmentMag::bestCand(/*inputs*/
const MultidimArray<double> &MDaIn,
		const MultidimArray<std::complex<double> > &MDaInF,
		const MultidimArray<double> &MDaRef, std::vector<double> &cand,
		/*outputs*/
		double &psi, double &shift_x, double &shift_y, double &bestCoeff) {
	psi = 0;
	shift_x = 0.;
	shift_y = 0.;
	bestCoeff = 0.0;
	double rotVar = 0.0;
	double tempCoeff;
	double tx;
	double ty;
	MultidimArray<double> MDaRefRot;
	MultidimArray<double> MDaRefShiftRot;
	MultidimArray<double> ccMatrixShift;
	MultidimArray<double> ccVectorTx;
	MultidimArray<double> ccVectorTy;
	MultidimArray<std::complex<double> > MDaRefRotF;

	MDaRefRot.setXmippOrigin();
	MDaRefShiftRot.setXmippOrigin();

	MultidimArray<double> MDaInShift;
	MultidimArray<double> MDaInShiftRot;
	MDaInShift.setXmippOrigin();
	MDaInShiftRot.setXmippOrigin();

	for (int i = 0; i < peaksFound; ++i) {
		rotVar = -1. * cand[i];  //negative, because is for reference rotation
		applyRotation(MDaRef, rotVar, MDaRefRot); //rotation to reference image
		applyFourierImage2(MDaRefRot, MDaRefRotF); //Fourier
		ccMatrix(MDaInF, MDaRefRotF, ccMatrixShift); // cross-correlation matrix
/*comment
testCounter+=1;
if(testCounter==2){
std::cout<<"size of ccMatrixShift in bestCand:"<<YSIZE(ccMatrixShift)<<"x"<<XSIZE(ccMatrixShift)<<"\n";
} // */

		maxByColumn(ccMatrixShift, ccVectorTx); // ccvMatrix to ccVector
		getShift(ccVectorTx, tx, XSIZE(ccMatrixShift));
		tx = -1. * tx;
		maxByRow(ccMatrixShift, ccVectorTy); // ccvMatrix to ccVector
		getShift(ccVectorTy, ty, YSIZE(ccMatrixShift));
		ty = -1. * ty;

		if (std::abs(tx) > maxShift || std::abs(ty) > maxShift)
			continue;

		//apply transformation to experimental image
		double expTx;
		double expTy;
		double expPsi;
		expPsi = -rotVar;
		expTx = -tx;
		expTy = -ty;
		// applying in one transform
		applyShiftAndRotation(MDaIn, expPsi, expTx, expTy, MDaInShiftRot);

		circularWindow(MDaInShiftRot); //circular masked MDaInRotShift

		// distance
		pearsonCorr(MDaRef, MDaInShiftRot, tempCoeff);  // Pearson
		//		normalized_cc(MDaRef, MDaInShiftRot, tempCoeff); // NCC
		//		imNormalized_cc(MDaRef, MDaInShiftRot, tempCoeff); // IMNCC
		//		imZNCC(MDaRef, MDaInShiftRot, tempCoeff); // IMZNCC
		if (tempCoeff > bestCoeff) {
			bestCoeff = tempCoeff;
			shift_x = -expTx; //negative because in second loop,when used, this parameters are applied to mdaRef
			shift_y = -expTy;
			psi = -expPsi;
		}
	}
}

/* same as bestCand method but using 2 candidates for shift instead of 1 only
 */
void ProgAngularAssignmentMag::bestCand2(/*inputs*/
const MultidimArray<double> &MDaIn,
		const MultidimArray<std::complex<double> > &MDaInF,
		const MultidimArray<double> &MDaRef, std::vector<double> &cand,
		/*outputs*/
		double &psi, double &shift_x, double &shift_y, double &bestCoeff) {
	psi = 0;
	shift_x = 0.;
	shift_y = 0.;
	bestCoeff = 0.0;
	double rotVar = 0.0;
	double tempCoeff;

	MultidimArray<double> MDaRefRot;
	MultidimArray<double> MDaRefShiftRot;
	MultidimArray<double> ccMatrixShift;
	MultidimArray<double> ccVectorTx;
	MultidimArray<double> ccVectorTy;
	MultidimArray<std::complex<double> > MDaRefRotF;

	MDaRefRot.setXmippOrigin();
	MDaRefShiftRot.setXmippOrigin();

	MultidimArray<double> MDaInShift;
	MultidimArray<double> MDaInShiftRot;
	MDaInShift.setXmippOrigin();
	MDaInShiftRot.setXmippOrigin();

	for (int i = 0; i < peaksFound; ++i) {
		rotVar = -1. * cand[i];  //negative, because is for reference rotation
		applyRotation(MDaRef, rotVar, MDaRefRot); //rotation to reference image
		applyFourierImage2(MDaRefRot, MDaRefRotF); //fourier
		ccMatrix(MDaInF, MDaRefRotF, ccMatrixShift); // cross-correlation matrix

		// testing 2 candidates for each psi
		std::vector<double> vTx(2, 0.);
		std::vector<double> vTy(2, 0.);
		maxByColumn(ccMatrixShift, ccVectorTx); // ccvMatrix to ccVector
		getShift2(ccVectorTx, vTx, XSIZE(ccMatrixShift));
		maxByRow(ccMatrixShift, ccVectorTy); // ccvMatrix to ccVector
		getShift2(ccVectorTy, vTy, YSIZE(ccMatrixShift));

		//      // In case this approach works properly, then set a condition to "continue"
		//		if (std::abs(vTx[0]) > maxShift || std::abs(vTy[0]) > maxShift ||
		//				std::abs(vTx[1]) > maxShift || std::abs(vTy[1]) > maxShift)
		//			continue;

		double expTx;
		double expTy;
		double expPsi;
		expPsi = -rotVar;
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 2; ++k) {
				//apply transformation to experimental image
				expTx = vTx[j];
				expTy = vTy[k];
				// applying in one transform
				applyShiftAndRotation(MDaIn, expPsi, expTx, expTy,
						MDaInShiftRot);
				circularWindow(MDaInShiftRot); //circular masked MDaInRotShift
				pearsonCorr(MDaRef, MDaInShiftRot, tempCoeff);  // Pearson
				if (tempCoeff > bestCoeff) {
					bestCoeff = tempCoeff;
					shift_x = -expTx; //negative because in second loop,when used, this parameters are applied to mdaRef
					shift_y = -expTy;
					psi = -expPsi;
				}
			}
		}
	}
}


/* apply rotation */
void ProgAngularAssignmentMag::applyRotation(const MultidimArray<double> &MDaRef, double &rot,
		MultidimArray<double> &MDaRefRot) {
	// Transform matrix
	Matrix2D<double> A(3, 3);
	A.initIdentity();
	double ang;
	double cosine;
	double sine;
	ang = DEG2RAD(rot);
	cosine = cos(ang);
	sine = sin(ang);

	// rotation
	MAT_ELEM(A,0, 0) = cosine;
	MAT_ELEM(A,0, 1) = sine;
	MAT_ELEM(A,1, 0) = -sine;
	MAT_ELEM(A,1, 1) = cosine;

	// Shift
	MAT_ELEM(A,0, 2) = 0.;
	MAT_ELEM(A,1, 2) = 0.;

	applyGeometry(LINEAR, MDaRefRot, MDaRef, A, IS_NOT_INV, DONT_WRAP);
}

/* apply rotation */
void ProgAngularAssignmentMag::applyRotation(MultidimArray<double> &MDaRef,
		double &rot, MultidimArray<double> &MDaRefRot) {
	// Transform matrix
	Matrix2D<double> A(3, 3);
	A.initIdentity();
	double ang;
	double cosine;
	double sine;
	ang = DEG2RAD(rot);
	cosine = cos(ang);
	sine = sin(ang);

	// rotation
	MAT_ELEM(A,0, 0) = cosine;
	MAT_ELEM(A,0, 1) = sine;
	MAT_ELEM(A,1, 0) = -sine;
	MAT_ELEM(A,1, 1) = cosine;

	// Shift
	MAT_ELEM(A,0, 2) = 0.;
	MAT_ELEM(A,1, 2) = 0.;

	applyGeometry(LINEAR, MDaRefRot, MDaRef, A, IS_NOT_INV, DONT_WRAP);
}

/* apply translation */
void ProgAngularAssignmentMag::applyShift(MultidimArray<double> &input,
		double &tx, double &ty, MultidimArray<double> &output) {
	// Transform matrix
	Matrix2D<double> A(3, 3);
	A.initIdentity();

	// Shift
	MAT_ELEM(A,0, 2) = tx;
	MAT_ELEM(A,1, 2) = ty;

	applyGeometry(LINEAR, output, input, A, IS_NOT_INV, DONT_WRAP);
}

/* apply translation */
void ProgAngularAssignmentMag::applyShift(const MultidimArray<double> &input,
		double &tx, double &ty, MultidimArray<double> &output) {
	// Transform matrix
	Matrix2D<double> A(3, 3);
	A.initIdentity();

	// Shift
	MAT_ELEM(A,0, 2) = tx;
	MAT_ELEM(A,1, 2) = ty;

	applyGeometry(LINEAR, output, input, A, IS_NOT_INV, DONT_WRAP);
}

/* finds shift as maximum of ccVector */
void ProgAngularAssignmentMag::getShift(MultidimArray<double> &ccVector,
		double &shift, const size_t &size) {
	double maxVal = -10.;
	int idx = 0;
	int lb = int(size / 2 - maxShift);
	int hb = int(size / 2 + maxShift);
	for (int i = lb; i < hb; ++i) { //i= lb : hb or i= 1: size-1
		if (    ( dAi(ccVector,size_t(i)) > dAi(ccVector, size_t(i - 1))) &&
				( dAi(ccVector,size_t(i)) > dAi(ccVector, size_t(i + 1))) && // is this value a peak value?
				(dAi(ccVector,i) > maxVal) ) {  // is the biggest?
			maxVal = dAi(ccVector, i);
			idx = i;
		}
	}

	if (idx) {
		// interpolate value
		double interpIdx;
		interpIdx = quadInterp(idx, ccVector);
		shift = double(size) / 2. - interpIdx;
	} else {
		shift = maxShift+1;
	}

}

/* finds shift as maximum of ccVector */
void ProgAngularAssignmentMag::getShift2(MultidimArray<double> &ccVector,
		std::vector<double> &cand, const size_t &size) {
	double max1 = -1000.;
	int idx1 = 0;
	double max2 = -1000.;
	int idx2 = 0;

	for (int i = 1; i < size - 1; ++i) { // only look for in range 90:-90
		// current value is a peak value?
		if ((dAi(ccVector,size_t(i)) > dAi(ccVector, size_t(i - 1)))
				&& (dAi(ccVector,size_t(i)) > dAi(ccVector, size_t(i + 1)))) {
			if ( dAi(ccVector,i) > max1) {
				max2 = max1;
				idx2 = idx1;
				max1 = dAi(ccVector, i);
				idx1 = i;
			} else if ( dAi(ccVector,i) > max2 && dAi(ccVector,i) != max1) {
				max2 = dAi(ccVector, i);
				idx2 = i;
			}
		}
	}
	if (idx1 != 0) {
		int maxAccepted = 1;
		std::vector<int> temp;
		if (idx2 != 0) {
			maxAccepted = 2;
			temp.resize(maxAccepted);
			temp[0] = idx1;
			temp[1] = idx2;
		} else {
			temp.resize(maxAccepted);
			temp[0] = idx1;
		}

		double interpIdx; // quadratic interpolated location of peak
		for (int i = 0; i < maxAccepted; ++i) {
			interpIdx = quadInterp(temp[i], ccVector);
			cand[i] = double(size) / 2. - interpIdx;
		}
	} else {
		std::cout << "no peaks!\n";
	}
}

/* Structural similarity SSIM index Coeff */
void ProgAngularAssignmentMag::ssimIndex(MultidimArray<double> &X,
		MultidimArray<double> &Y, double &coeff) {

	// covariance
	double X_m;
	double Y_m;
	double X_std;
	double Y_std;
	double c1;
	double c2;
	double L;
	arithmetic_mean_and_stddev(X, X_m, X_std);
	arithmetic_mean_and_stddev(Y, Y_m, Y_std);

	double prod_mean = mean_of_products(X, Y);
	double covariace = prod_mean - (X_m * Y_m);

	L = 1;
	c1 = (0.01 * L) * (0.01 * L);
	c2 = (0.03 * L) * (0.03 * L); // for stability

	coeff = ((2 * X_m * Y_m + c1) * (2 * covariace + c2))
			/ ((X_m * X_m + Y_m * Y_m + c1) * (X_std * X_std + Y_std * Y_std + c2));
}

/* Structural similarity SSIM index Coeff */
void ProgAngularAssignmentMag::ssimIndex(const MultidimArray<double> &X,
		MultidimArray<double> &Y, double &coeff) {

	// covariance
	double X_m;
	double Y_m;
	double X_std;
	double Y_std;
	double c1;
	double c2;
	double L;
	arithmetic_mean_and_stddev(X, X_m, X_std);
	arithmetic_mean_and_stddev(Y, Y_m, Y_std);

	double prod_mean = mean_of_products(X, Y);
	double covariace = prod_mean - (X_m * Y_m);

	L = 1;
	c1 = (0.01 * L) * (0.01 * L);
	c2 = (0.03 * L) * (0.03 * L); // stability in division

	coeff = ((2 * X_m * Y_m + c1) * (2 * covariace + c2))
			/ ((X_m * X_m + Y_m * Y_m + c1) * (X_std * X_std + Y_std * Y_std + c2));
}

/* apply rotation then shift */
void ProgAngularAssignmentMag::applyRotationAndShift(const MultidimArray<double> &MDaRef, double &rot, double &tx,
		double &ty, MultidimArray<double> &MDaRefRot) {
	// Transform matrix
	Matrix2D<double> A(3, 3);
	A.initIdentity();
	double ang;
	double cosine;
	double sine;
	ang = DEG2RAD(rot);
	cosine = cos(ang);
	sine = sin(ang);

	// rotation
	MAT_ELEM(A,0, 0) = cosine;
	MAT_ELEM(A,0, 1) = sine;
	MAT_ELEM(A,1, 0) = -sine;
	MAT_ELEM(A,1, 1) = cosine;

	// Shift
	MAT_ELEM(A,0, 2) = tx;
	MAT_ELEM(A,1, 2) = ty;

	applyGeometry(LINEAR, MDaRefRot, MDaRef, A, IS_NOT_INV, DONT_WRAP);
}

/* apply shift then rotation */
void ProgAngularAssignmentMag::applyShiftAndRotation(const MultidimArray<double> &MDaRef, double &rot, double &tx,
		double &ty, MultidimArray<double> &MDaRefRot) {
	// Transform matrix
	Matrix2D<double> A(3, 3);
	A.initIdentity();
	double ang;
	double cosine;
	double sine;
	ang = DEG2RAD(rot);
	cosine = cos(ang);
	sine = sin(ang);

	// rotate in opposite direction
	double realTx = cosine * tx + sine * ty;
	double realTy = -sine * tx + cosine * ty;

	// rotation
	MAT_ELEM(A,0, 0) = cosine;
	MAT_ELEM(A,0, 1) = sine;
	MAT_ELEM(A,1, 0) = -sine;
	MAT_ELEM(A,1, 1) = cosine;

	// Shift
	MAT_ELEM(A,0, 2) = realTx;
	MAT_ELEM(A,1, 2) = realTy;

	applyGeometry(LINEAR, MDaRefRot, MDaRef, A, IS_NOT_INV, DONT_WRAP);
}

