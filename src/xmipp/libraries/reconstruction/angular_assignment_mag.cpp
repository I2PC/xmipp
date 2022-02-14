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
#include <core/metadata_extension.h>
#include <core/utils/memory_utils.h>
#include "data/projection.h"
#include "data/fourier_projection.h"
#include <reconstruction/project_real_shears.h>
#include <mutex>

#include <fstream>
#include <ctime>
#include <unistd.h>

ProgAngularAssignmentMag::ProgAngularAssignmentMag(){
	produces_a_metadata = true;
	each_image_produces_an_output = false;
	rank=0;
	Nprocessors=1;
}

ProgAngularAssignmentMag::~ProgAngularAssignmentMag()  = default;

void ProgAngularAssignmentMag::defineParams() {
	XmippMetadataProgram::defineParams();
	//usage
	addUsageLine( "Generates a list of candidates for angular assignment for each experimental image");
	addParamsLine("   -ref <md_file>             : Metadata file with input reference projections");
	addParamsLine("  [-odir <outputDir=\".\">]   : Output directory");
	addParamsLine("  [--sym <symfile=c1>]         : Enforce symmetry in projections");
	addParamsLine("  [-sampling <sampling=1.>]   : sampling");
	addParamsLine("  [-angleStep <angStep=3.>]   : angStep");
	addParamsLine("  [--maxShift <maxShift=-1.>]  : Maximum shift allowed (+-this amount)");
	addParamsLine("  [--Nsimultaneous <Nprocessors=1>]  : Nsimultaneous");
	addParamsLine("  [--refVol <refVolFile=NULL>]  : reference volume to be reprojected when comparing with previous alignment");
	addParamsLine("  [--useForValidation] : Use the program for validation");
	addParamsLine("  [--thr <threads=4>]           : How many threads to use");
}

// Read arguments ==========================================================
void ProgAngularAssignmentMag::readParams() {
	XmippMetadataProgram::readParams();
	fnRef = getParam("-ref");
	fnDir = getParam("-odir");
	sampling = getDoubleParam("-sampling");
	angStep= getDoubleParam("-angleStep");
	XmippMetadataProgram::oroot = fnDir;
	fnSym = getParam("--sym");
	maxShift = getDoubleParam("--maxShift");
	inputReference_volume = getParam("--refVol");
	useForValidation=checkParam("--useForValidation");
	threads = getIntParam("--thr");
	threadPool.resize(threads);
	transformersForImages.resize(threads);
	ccMatrixBestCandidTransformers.resize(threads);
	ccMatrixShifts.resize(threads);
}

// Show ====================================================================
void ProgAngularAssignmentMag::show() const {
	if (verbose > 0) {
		printf("%d reference images of %d x %d\n",int(sizeMdRef),int(Xdim),int(Ydim));
		printf("%d exp images of %d x %d in this group\n", int(sizeMdIn),int(Xdim), int(Ydim));
		std::cout << "Sampling: " << sampling << std::endl;
		std::cout << "Angular step: " << angStep << std::endl;
		std::cout << "Maximum shift: " << maxShift << std::endl;
		std::cout << "threads: " << threads << std::endl;
		if(useForValidation){
			std::cout << "ref vol size: " << refXdim <<" x "<< refYdim <<" x "<< refZdim << std::endl;
			std::cout << "useForValidation            : "  << useForValidation << std::endl;
		}
		XmippMetadataProgram::show();
		std::cout << "Input references: " << fnRef << std::endl;
	}
}

/*
 * In this method, for each direction, I look for neighbors within some distance
 * */
void ProgAngularAssignmentMag::computingNeighborGraph() {
	std::vector< std::vector<int> > allNeighborsjp;
	std::vector< std::vector<double> > allWeightsjp;
	Matrix1D<double> distanceToj;
	Matrix1D<double> dirj;
	Matrix1D<double> dirjp;
	double maxSphericalDistance = angStep*2.;
	std::cout<< "processing neighbors graph..."<<std::endl;

	for (auto &rowj : mdRef){
		double rotj;
		double tiltj;
		double psij;
		rowj.getValue(MDL_ANGLE_ROT, rotj);
		rowj.getValue(MDL_ANGLE_TILT, tiltj);
		rowj.getValue(MDL_ANGLE_PSI, psij);
		distanceToj.initZeros(sizeMdRef);
		Euler_direction(rotj, tiltj, psij, dirj);
		std::vector<int> neighborsjp;
		std::vector<double> weightsjp;
		double thisSphericalDistance = 0.;
		int jp = 0;
		for (auto &rowjp : mdRef){
			double rotjp;
			double tiltjp;
			double psijp;
			rowjp.getValue(MDL_ANGLE_ROT, rotjp);
			rowjp.getValue(MDL_ANGLE_TILT, tiltjp);
			rowjp.getValue(MDL_ANGLE_PSI, psijp);
			Euler_direction(rotjp, tiltjp, psijp, dirjp);
			thisSphericalDistance = RAD2DEG(spherical_distance(dirj, dirjp));

			if (thisSphericalDistance < maxSphericalDistance) {
				neighborsjp.emplace_back(jp);
				double val = exp(-thisSphericalDistance / maxSphericalDistance);
				weightsjp.emplace_back(val);
			}
			jp++;
		}
		allNeighborsjp.emplace_back(neighborsjp);
		allWeightsjp.emplace_back(weightsjp);
	}// END FOR_ALL_OBJECTS_IN_METADATA(mdRef)

	// compute Laplacian Matrix
	DMatrix L_mat;
	computeLaplacianMatrix(L_mat, allNeighborsjp, allWeightsjp);

	// from diagSymMatrix3x3 method in resolution_directional.cpp
	Matrix2D<double> B;
	B.resizeNoCopy(L_mat);
	B.initIdentity();
	generalizedEigs(L_mat, B, eigenvalues, eigenvectors);

	// save eigenvalues y eigenvectors files
	String fnEigenVal = formatString("%s/outEigenVal.txt", fnDir.c_str());
	eigenvalues.write(fnEigenVal);
	String fnEigenVect = formatString("%s/outEigenVect.txt", fnDir.c_str());
	eigenvectors.write(fnEigenVect);
}

/* Laplacian Matrix is basic for signal graph processing stage
 * is computed only once within preProcess() method */
void ProgAngularAssignmentMag::computeLaplacianMatrix (Matrix2D<double> &matL,
		const std::vector< std::vector<int> > &allNeighborsjp,
		const std::vector< std::vector<double> > &allWeightsjp) const {

	matL.initZeros(sizeMdRef,sizeMdRef);

	for(int i=0; i<sizeMdRef; ++i){
		std::vector<int> neighborsjp = allNeighborsjp[i];
		std::vector<double> weightsjp = allWeightsjp[i];
		double sumWeight = 0.;
		int indx = 0;
		int j = -1;
		for(auto it=neighborsjp.begin(); it!=neighborsjp.end(); ++it){
			j += 1;
			indx = (*it);
			MAT_ELEM(matL,i,indx) = -weightsjp[j];
			sumWeight += weightsjp[j];
		}
		MAT_ELEM(matL,i,i) = sumWeight - 1.; // -1 because is necessary to remove the "central" weight
	}
}

unsigned long long getTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}

void ProgAngularAssignmentMag::preProcess() {

	mdIn.read(XmippMetadataProgram::fn_in);
	mdRef.read(fnRef);

	// size of images
	size_t Zdim;
	size_t Ndim;
	getImageSize(mdIn, Xdim, Ydim, Zdim, Ndim);

	// some constants
	n_rad = size_t((double)Xdim / 2.);
	startBand = size_t((sampling * (double) Xdim) / 80.);
	finalBand = size_t((sampling * (double) Xdim) / (sampling * 3));
	n_bands = finalBand - startBand;
	n_ang = size_t(180);
	n_ang2 = 2 * n_ang;
	if (maxShift==-1.){
		maxShift = .10 * (double)Xdim;
	}

	// read reference images
	FileName fnImgRef;
	sizeMdRef = (int)mdRef.size();

	// how many input images
	sizeMdIn = (int)mdIn.size();

	// reference image related
	Image<double> ImgRef;
	MultidimArray<double> MDaRef((int)Ydim, (int)Xdim);
	MultidimArray<std::complex<double> > MDaRefF;
	MultidimArray<std::complex<double> > MDaRefF2;
	MultidimArray<double> MDaRefFM;
	MultidimArray<double> MDaRefFMs;
	MultidimArray<double> MDaRefFMs_polarPart((int)n_bands, (int)n_ang2);
	MultidimArray<std::complex<double> > MDaRefFMs_polarF;

	MultidimArray<double> refPolar((int)n_rad, (int)n_ang2);
	MultidimArray<std::complex<double> > MDaRefAuxF;

	computeCircular(); //pre-compute circular mask

	// for storage of rot and tilt of reference images
	referenceRot.resize(sizeMdRef);
	referenceTilt.resize(sizeMdRef);
	if (rank == 0) {
		std::cout << "processing reference library..." << std::endl;
		// memory check
		size_t dataSize = Xdim * Ydim * sizeMdRef * sizeof(double);
		size_t matrixSize = sizeMdRef * sizeMdRef * sizeof(double);
		size_t polarFourierSize = n_bands * n_ang2 * sizeMdRef * sizeof(std::complex<double>);

		size_t totMemory = dataSize + matrixSize + polarFourierSize;
		totMemory = memoryUtils::MB(totMemory) * Nprocessors;
		std::cout << "approx. memory to allocate: " << totMemory << " MB" << std::endl;
		std::cout << "simultaneous MPI processes: " << Nprocessors << std::endl;

		size_t available = getTotalSystemMemory();
		available = memoryUtils::MB(available);
		std::cout << "total available system memory: " << available << " MB" << std::endl;
		available -= 1500;

	    if (available < totMemory ) {
	        REPORT_ERROR(ERR_MEM_NOTENOUGH,"You don't have enough memory. Try to use less MPI processes.");
	    }
	}

	int j = 0;
	for (size_t objId : mdRef.ids()){
		// reading image
		mdRef.getValue(MDL_IMAGE, fnImgRef, objId);
		ImgRef.read(fnImgRef);
		MDaRef = ImgRef();
		MDaRef.setXmippOrigin();
		// store to call in processImage method
		double rot;
		double tilt;
		double psi;
		mdRef.getValue(MDL_ANGLE_ROT, rot, objId);
		mdRef.getValue(MDL_ANGLE_TILT, tilt, objId);
		mdRef.getValue(MDL_ANGLE_PSI, psi, objId);
		referenceRot.at(j) = rot;
		referenceTilt.at(j) = tilt;
		// processing reference image
		vecMDaRef.push_back(MDaRef);
		applyFourierImage2(MDaRef, MDaRefF);
		// Fourier of polar magnitude spectra
		transformerImage.getCompleteFourier(MDaRefF2);
		getComplexMagnitude(MDaRefF2, MDaRefFM);
		completeFourierShift(MDaRefFM, MDaRefFMs);
		MDaRefFMs_polarPart = imToPolar(MDaRefFMs, startBand, finalBand);
		applyFourierImage2(MDaRefFMs_polarPart, MDaRefFMs_polarF, n_ang);
		vecMDaRefFMs_polarF.push_back(MDaRefFMs_polarF);
		j++;
	}

	// check if eigenvectors file already created
	String fnEigenVect = formatString("%s/outEigenVect.txt",fnDir.c_str());
	std::ifstream in;
	in.open(fnEigenVect.c_str());
	if(!in){
		in.close();
		// Define the neighborhood graph, Laplacian Matrix and eigendecomposition
		if(rank == 0)
			computingNeighborGraph();
	}

	// synch with other processors
	synchronize();

	// prepare and read from file
	eigenvectors.clear();
	eigenvectors.resizeNoCopy(sizeMdRef, sizeMdRef);
	eigenvectors.read(fnEigenVect);

	// Symmetry List
	if (fnSym != "") {
		SL.readSymmetryFile(fnSym);
		for (int sym = 0; sym < SL.symsNo(); sym++) {
			Matrix2D<double> auxL, auxR;
			SL.getMatrices(sym, auxL, auxR);
			auxL.resize(3, 3);
			auxR.resize(3, 3);
			L.push_back(auxL);
			R.push_back(auxR);
		}
	} // */

	if(useForValidation){
		// read reference volume to be re-projected when comparing previous assignment
		// If there is no reference available, exit
		try{
			refVol.read(inputReference_volume);
		}
		catch (XmippError &XE){
			std::cout << XE;
			exit(0);
		}
		refVol().setXmippOrigin();
		refXdim = (int)XSIZE(refVol());
		refYdim = (int)YSIZE(refVol());
		refZdim = (int)ZSIZE(refVol());
	}
}

/* Apply graph signal processing to cc-vector using the Laplacian eigen-decomposition
 * */
void ProgAngularAssignmentMag::graphFourierFilter(const Matrix1D<double> &ccVecIn, Matrix1D<double> &ccVecOut ) const {
	// graph signal processing filtered iGFT
	Matrix1D<double> ccGFT;
	ccGFT.initZeros(sizeMdRef);
	std::vector<double> filt_iGFT_cc(sizeMdRef, 0);

	Matrix2D<double> eigenvectorTrans = eigenvectors.transpose();
	ccGFT = eigenvectorTrans*ccVecIn;

	// define filtering base
	int cutEig = (sizeMdRef>1000) ? int(.05 * sizeMdRef + 1) : int(.50 * sizeMdRef + 1);
	for(int k = cutEig; k < sizeMdRef; ++k){
		VEC_ELEM(ccGFT,k) = 0.;
	}
	// apply filter to ccvec
	ccVecOut = eigenvectors*ccGFT;
}

void ProgAngularAssignmentMag::processImage(const FileName &fnImg,const FileName &fnImgOut,
		const MDRow &rowIn, MDRow &rowOut) {

	// experimental image related
	Image<double> ImgIn;
	MultidimArray<double> MDaIn((int)Ydim, (int)Xdim);
	MultidimArray<std::complex<double> > MDaInF;
	MultidimArray<std::complex<double> > MDaInF2;
	MultidimArray<double> MDaInFM;
	MultidimArray<double> MDaInFMs;
	MultidimArray<double> MDaInFMs_polarPart((int)n_bands, (int)n_ang2);
	MultidimArray<std::complex<double> > MDaInFMs_polarF;

	// processing input image
	ImgIn.read(fnImg);
	MDaIn = ImgIn();
	MDaIn.setXmippOrigin();
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

	std::vector<unsigned int> Idx(sizeMdRef, 0);

	Matrix1D<double> ccvec;
	ccvec.initZeros(sizeMdRef);

	std::vector<double> bestTx(sizeMdRef, 0);
	std::vector<double> bestTy(sizeMdRef, 0);
	std::vector<double> bestPsi(sizeMdRef, 0);

	MultidimArray<double> ccMatrixRot;
	MultidimArray<double> ccVectorRot;

	// loop over all reference images
//	 /*
	for (int k = 0; k < sizeMdRef; ++k) {
		// computing relative rotation and shift
		ccMatrix(MDaInFMs_polarF, vecMDaRefFMs_polarF[k], ccMatrixRot, ccMatrixProcessImageTransformer);
		maxByColumn(ccMatrixRot, ccVectorRot);
		peaksFound = 0;
		std::vector<double> cand(maxAccepted, 0.);
		psiCandidates(ccVectorRot, cand, XSIZE(ccMatrixRot));
		bestCand(MDaIn, MDaInF, vecMDaRef[k], cand, psi, Tx, Ty, cc_coeff);
		// all results are storage for posterior partial_sort
		Idx[k] = k; // for sorting
		VEC_ELEM(ccvec,k) = cc_coeff;
		bestTx[k] = Tx;
		bestTy[k] = Ty;
		bestPsi[k] = psi;
	}

	// variables for second loop
	std::vector<double> bestTx2(sizeMdRef, 0.);
	std::vector<double> bestTy2(sizeMdRef, 0.);
	std::vector<double> bestPsi2(sizeMdRef, 0.);

	MultidimArray<double> &MDaInAux = ImgIn();
	MDaInAux.setXmippOrigin();
	MultidimArray<double> mCurrentImageAligned;
	double corr, scale;
	bool flip;

	double minval, maxval;
	ccvec.computeMinMax(minval, maxval);
	double thres = 0.;
	if (SL.symsNo()<=4)
		thres = maxval - (maxval - minval) / 3.;
	else
		thres = maxval - (maxval - minval) / 2.;

	for(int k = 0; k < sizeMdRef; ++k) {
		if(VEC_ELEM(ccvec,k)>thres){
			// find rotation and shift using alignImages
			Matrix2D<double> M;
			mCurrentImageAligned = MDaInAux;
			mCurrentImageAligned.setXmippOrigin();
			corr = alignImages(vecMDaRef[k], mCurrentImageAligned, M, xmipp_transformation::DONT_WRAP);
			M = M.inv();
			transformationMatrix2Parameters2D(M, flip, scale, Tx, Ty, psi);

			if (maxShift>0 && (fabs(Tx)>maxShift || fabs(Ty)>maxShift))
				corr /= 3;

			VEC_ELEM(ccvec, k) = corr;
			bestTx2[k] = Tx;
			bestTy2[k] = Ty;
			bestPsi2[k] = psi;

		}
		else{
			bestTx2[k] = bestTx[k];
			bestTy2[k] = bestTy[k];
			bestPsi2[k] = bestPsi[k];
		}
	}

	// ================ Graph Filter Process after second loop =================
	Matrix1D<double> ccvec_filt;
	graphFourierFilter(ccvec,ccvec_filt);

	// choose best of the candidates after 2nd loop
	int idx = ccvec.maxIndex();

	// choose best candidate direction from graph filtered ccvect signal
	int idxfilt = ccvec_filt.maxIndex();

	// angular distance between this two directions
	Matrix1D<double> dirj;
	Matrix1D<double> dirjp;
	double rotj = referenceRot.at(idx);
	double tiltj = referenceTilt.at(idx);
	double psij = 0.;
	Euler_direction(rotj, tiltj, psij, dirj);
	double rotjp = referenceRot.at(idxfilt);
	double tiltjp = referenceTilt.at(idxfilt);
	double psijp = 0.;
	Euler_direction(rotjp, tiltjp, psijp, dirjp);
	double sphericalDistance = RAD2DEG(spherical_distance(dirj, dirjp));

	// compute distance keeping in mind the symmetry list
	for(size_t sym = 0; sym<SL.symsNo(); sym++){
		double auxRot, auxTilt, auxPsi; // equivalent coordinates
		double auxSphericalDist;
		Euler_apply_transf(L[sym], R[sym],rotjp, tiltjp, psijp,
				auxRot,auxTilt,auxPsi);
		Euler_direction(auxRot, auxTilt, auxPsi, dirjp);
		auxSphericalDist = RAD2DEG(spherical_distance(dirj, dirjp));
		if (auxSphericalDist < sphericalDistance)
			sphericalDistance = auxSphericalDist;
	}

	// set output alignment parameters values
	double rotRef = referenceRot.at(idx);
	double tiltRef = referenceTilt.at(idx);
	double shiftX = bestTx2[idx];
	double shiftY = bestTy2[idx];
	double anglePsi = bestPsi2[idx];
	corr = ccvec[idx];

	//save metadata of images with angles
	rowOut.setValue(MDL_IMAGE, fnImgOut);
	rowOut.setValue(MDL_ENABLED, 1);
	rowOut.setValue(MDL_MAXCC, corr);
	rowOut.setValue(MDL_ANGLE_ROT, rotRef);
	rowOut.setValue(MDL_ANGLE_TILT, tiltRef);
	rowOut.setValue(MDL_ANGLE_PSI, realWRAP(anglePsi, -180., 180.));
	rowOut.setValue(MDL_SHIFT_X, -shiftX);
	rowOut.setValue(MDL_SHIFT_Y, -shiftY);
	rowOut.setValue(MDL_WEIGHT, 1.);
	rowOut.setValue(MDL_WEIGHT_SIGNIFICANT, 1.);
	rowOut.setValue(MDL_GRAPH_DISTANCE2MAX, sphericalDistance);


	if(!useForValidation){
		// align & correlation between reference images located at idx and idxfilt
		Matrix2D<double> M2;
		double graphCorr = alignImages(vecMDaRef[idx], vecMDaRef[idxfilt], M2, xmipp_transformation::DONT_WRAP);
		rowOut.setValue(MDL_GRAPH_CC, graphCorr);
	}
	else{
		// assignment in this run
		// get projection of volume from coordinates computed using my method
		Projection P2;
		double initPsiAngle = 0.;
		projectVolume(refVol(), P2, refYdim, refXdim, rotRef, tiltRef, initPsiAngle);
		MultidimArray<double> projectedReference2((int)Ydim, (int)Xdim);
		projectedReference2 = P2();

		double filtRotRef = referenceRot.at(idxfilt);
		double filtTiltRef = referenceTilt.at(idxfilt);
		Projection P3;
		projectVolume(refVol(), P3, refYdim, refXdim, filtRotRef, filtTiltRef, initPsiAngle);
		MultidimArray<double> projectedReference3((int)Ydim, (int)Xdim);
		projectedReference3 = P3();

		// align & correlation between reference images located at idx and idxfilt
		Matrix2D<double> M2;
		double graphCorr = alignImages(projectedReference2, projectedReference3, M2, xmipp_transformation::DONT_WRAP);
		rowOut.setValue(MDL_GRAPH_CC, graphCorr);

		// related to previous assignment
		double old_rot, old_tilt, old_psi, old_shiftX, old_shiftY;
		rowIn.getValue(MDL_ANGLE_ROT, old_rot);
		rowIn.getValue(MDL_ANGLE_TILT, old_tilt);
		rowIn.getValue(MDL_ANGLE_PSI, old_psi);
		rowIn.getValue(MDL_SHIFT_X, old_shiftX);
		rowIn.getValue(MDL_SHIFT_Y, old_shiftY);

		// get projection of volume from this coordinates
		Projection P;
		projectVolume(refVol(), P, refYdim, refXdim, old_rot, old_tilt, initPsiAngle);
		MultidimArray<double> projectedReference((int)Ydim, (int)Xdim);
		projectedReference = P();

		// align & correlation between reference images both methods
		// projectedReference2 is from this assignment
		Matrix2D<double> M;
		projectedReference2 = P2();
		double refCorr = alignImages(projectedReference, projectedReference2, M, xmipp_transformation::DONT_WRAP);

		// align & correlation between reference images by previous assignment and idxfilt
		projectedReference = P();
		projectedReference3 = P3();
		graphCorr = alignImages(projectedReference, projectedReference3, M, xmipp_transformation::DONT_WRAP);

		MultidimArray<double> mdainShifted((int)Ydim, (int)Xdim);
		mdainShifted.setXmippOrigin();
		old_psi *= -1;
		applyShiftAndRotation(MDaIn, old_psi, old_shiftX, old_shiftY, mdainShifted);
		circularWindow(mdainShifted); //circular masked
		double prevCorr = correlationIndex(projectedReference, mdainShifted);

		//angular distance between idxfilt direction and the given by previous alignment
		Matrix1D<double> dirjpCopy;
		dirjpCopy = dirjp;
		Matrix1D<double> dirPrevious;
		double relPsi = 0.;
		Euler_direction(old_rot, old_tilt, relPsi, dirPrevious);
		double algorithmSD = RAD2DEG(spherical_distance(dirjpCopy, dirPrevious));

		// compute distance keeping in mind the symmetry list
		for(size_t sym = 0; sym<SL.symsNo(); sym++){
			// related to compare against previous assignment
			double auxRot2, auxTilt2, auxPsi2; // equivalent coordinates
			double auxSphericalDist2;
			Euler_apply_transf(L[sym], R[sym], old_rot, old_tilt, relPsi,
					auxRot2,auxTilt2,auxPsi2);
			Euler_direction(auxRot2, auxTilt2, auxPsi2, dirPrevious);
			auxSphericalDist2 = RAD2DEG(spherical_distance(dirPrevious, dirjpCopy));
			if (auxSphericalDist2 < algorithmSD)
				algorithmSD = auxSphericalDist2;

		}

		rowOut.setValue(MDL_MAXCC_PREVIOUS, prevCorr);
		rowOut.setValue(MDL_GRAPH_DISTANCE2MAX_PREVIOUS, algorithmSD);
		rowOut.setValue(MDL_GRAPH_CC_PREVIOUS, graphCorr);
		rowOut.setValue(MDL_ASSIGNED_DIR_REF_CC, refCorr);
	}
}

void ProgAngularAssignmentMag::postProcess() {
	// from angularContinousAssign2
	MetaData &ptrMdOut = getOutputMd();

	ptrMdOut.removeDisabled();
	double maxCC = -1.;
	int j = 0;
	for (size_t objId : ptrMdOut.ids())
	{
		double thisMaxCC;
		ptrMdOut.getValue(MDL_MAXCC, thisMaxCC, objId);
		if (thisMaxCC > maxCC)
			maxCC = thisMaxCC;
		if (thisMaxCC == 0.0)
			ptrMdOut.removeObject(objId);
		j++;
	}
	for (size_t objId : ptrMdOut.ids())
	{
		double thisMaxCC;
		ptrMdOut.getValue(MDL_MAXCC, thisMaxCC, objId);
		ptrMdOut.setValue(MDL_WEIGHT, thisMaxCC / maxCC, objId);
		ptrMdOut.setValue(MDL_WEIGHT_SIGNIFICANT, thisMaxCC / maxCC,
				objId);
	}
	ptrMdOut.write(XmippMetadataProgram::fn_out.replaceExtension("xmd"));
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
	transformerPolarImage.FourierTransform(data, FourierData, true); // false --> true, to make a copy
}

/* get magnitude of fourier spectrum */
void ProgAngularAssignmentMag::getComplexMagnitude(
		const MultidimArray<std::complex<double> > &FourierData,
		MultidimArray<double> &FourierMag) const {
	FFT_magnitude(FourierData, FourierMag);
}

/* cartImg contains cartessian  grid representation of image,
 *  rad and ang are the number of radius and angular elements*/
MultidimArray<double> ProgAngularAssignmentMag::imToPolar(
		MultidimArray<double> &cartIm, size_t &start, size_t &end) {

	size_t thisNbands = end - start;
	MultidimArray<double> polarImg((int)thisNbands, (int)n_ang2);
	auto pi = (double)3.141592653;
	// coordinates of center
	double cy = (double)(Ydim + (Ydim % 2)) / 2.0; // for odd/even images
	double cx = (double)(Xdim + (Xdim % 2)) / 2.0;

	// scale factors
	double sfy = (double)(Ydim - 1) / 2.0;
	double sfx = (double)(Xdim - 1) / 2.0;

	auto delR = 1.0 / ((double)n_rad); // n_rad-1
	double delT = 2.0 * pi / ((double)n_ang2);

	// loop through rad and ang coordinates
	double r;
	double t;
	double x_coord;
	double y_coord;
	for (size_t ri = start; ri < end; ++ri) {
		for (size_t ti = 0; ti < n_ang2; ++ti) {
			r = (double)ri * delR;
			t = (double)ti * delT;
			x_coord = (r * cos(t)) * sfx + cx;
			y_coord = (r * sin(t)) * sfy + cy;
			// set value of polar img
			DIRECT_A2D_ELEM(polarImg,ri-start,ti) = interpolate(cartIm,x_coord,y_coord);
		}
	}
	return polarImg;
}

/* bilinear interpolation */
double ProgAngularAssignmentMag::interpolate(MultidimArray<double> &cartIm,
		double &x_coord, double &y_coord) const{
	double val;
	auto xf = (size_t)floor((double)x_coord);
	auto xc = (size_t)ceil((double)x_coord);
	auto yf = (size_t)floor((double)y_coord);
	auto yc = (size_t)ceil((double)y_coord);

	if ((xf == xc) && (yf == yc)) {
		val = dAij(cartIm, xc, yc);
	} else if (xf == xc) { // linear
		val = dAij(cartIm, xf, yf)
				+ (y_coord - (double) yf)
						* ( dAij(cartIm, xf, yc) - dAij(cartIm, xf, yf));
	} else if (yf == yc) { // linear
		val = dAij(cartIm, xf, yf)
				+ (x_coord - (double) xf)
						* ( dAij(cartIm, xc, yf) - dAij(cartIm, xf, yf));
	} else { // bilinear
		val = ((double) (( dAij(cartIm,xf,yf) * ((double) yc - y_coord)
				+ dAij(cartIm,xf,yc) * (y_coord - (double) yf))
				* ((double) xc - x_coord))
				+ (double) (( dAij(cartIm,xc,yf) * ((double) yc - y_coord)
						+ dAij(cartIm,xc,yc) * (y_coord - (double) yf))
						* (x_coord - (double) xf)))
				/ (double) ((xc - xf) * (yc - yf));
	}

	return val;
}

/* implementing centering using circular-shift*/
void ProgAngularAssignmentMag::completeFourierShift(const MultidimArray<double> &in,
		MultidimArray<double> &out) const {

	// correct output size
	out.resizeNoCopy(in);

	auto Cf = (size_t) ((double)YSIZE(in) / 2.0 + 0.5);
	auto Cc = (size_t) ((double)XSIZE(in) / 2.0 + 0.5);

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

/*
 * experiment for cross-correlation matrix product F1 .* conj(F2)
 */
void ProgAngularAssignmentMag::ccMatrix(const MultidimArray<std::complex<double>> &F1,
		const MultidimArray<std::complex<double>> &F2,/*reference image*/
		MultidimArray<double> &result,
		    FourierTransformer &transformer) {

	result.resizeNoCopy(YSIZE(F1), 2 * (XSIZE(F1) - 1));

	transformer.setReal(result);
	transformer.setFourier(F1);
	// Multiply FFT1 .* FFT2'
	double a;
	double b;
	double c;
	double d; // a+bi, c+di
	auto dSize = (double)MULTIDIM_SIZE(result);

	double *ptrFFT2 = (double*) MULTIDIM_ARRAY(F2);
	double *ptrFFT1 = (double*) MULTIDIM_ARRAY(transformer.fFourier);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F1)
	{
		a = (*ptrFFT1)* dSize;
		b = (*(ptrFFT1 + 1))* dSize;
		c = (*ptrFFT2++);
		d = (*ptrFFT2++) * (-1);
		*ptrFFT1++ = (a * c - b * d);
		*ptrFFT1++ = (b * c + a * d);
	}
	transformer.inverseFourierTransform();
	CenterFFT(result, true);
	result.setXmippOrigin();
}

/* gets maximum value for each column*/
void ProgAngularAssignmentMag::maxByColumn(const MultidimArray<double> &in,
		MultidimArray<double> &out) const {

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

/* gets maximum value for each row */
void ProgAngularAssignmentMag::maxByRow(const MultidimArray<double> &in,
		MultidimArray<double> &out) const {
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

/*quadratic interpolation for location of peak in crossCorr vector*/
double quadInterp(/*const*/int idx, const MultidimArray<double> &in) {

	double InterpIdx = idx	- ( ( dAi(in,idx+1) - dAi(in, idx - 1))
					/ ( dAi(in,idx+1) + dAi(in, idx - 1) - 2 * dAi(in, idx)) )
					/ 2.;
	return InterpIdx;
}

/* precompute circular 2D window Ydim x Xdim*/
void ProgAngularAssignmentMag::computeCircular() {

	auto Cf = (double)(Ydim + (Ydim % 2)) / 2.0; // for odd/even images
	auto Cc = (double)(Xdim + (Xdim % 2)) / 2.0;
	int pixReduc = 1;
	double rad2 = (Cf - pixReduc) * (Cf - pixReduc);
	double val = 0;

	C.resizeNoCopy(Ydim, Xdim);
	C.initZeros((int)Ydim, (int)Xdim);
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(C)
	{
		val = ((double)j - Cf) * ((double)j - Cf) + ((double)i - Cc) * ((double)i - Cc);
		if (val < rad2)
			dAij(C,i,j) = 1.;
	}
}

/*apply circular window to input image*/
void ProgAngularAssignmentMag::circularWindow(MultidimArray<double> &in) const {
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(in)
	{
		dAij(in,i,j) *= dAij(C, i, j);
	}
}

/* Only for 180 angles
 * just two locations of maximum peaks in ccvRot */
void ProgAngularAssignmentMag::psiCandidates(const MultidimArray<double> &in,
		std::vector<double> &cand, const size_t &size) {
	double max1 = -1000.;
	int idx1 = 0;
	double max2 = -1000.;
	int idx2 = 0;
	int cont = 0;
	peaksFound = cont;

	for (int i = 89; i < 272; ++i) { // only look within  90:-90 range
		// current value is a peak value?
		if ((dAi(in,size_t(i)) > dAi(in, size_t(i - 1)))
				&& (dAi(in,size_t(i)) > dAi(in, size_t(i + 1)))) {
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
			cand[i + maxAccepted] = (cand[i] >= 0.0) ? cand[i] + 180. : cand[i] - 180.;
		}
	} else {
		peaksFound = 0;
	}
}

/* selection of best candidate to rotation and its corresponding shift
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
	std::mutex mutex;

	auto workload = [&](int id, int i) {
		auto rotVar = -1. * cand[i];  //negative, because is for reference rotation
		MultidimArray<double> MDaRefRot;
		MDaRefRot.setXmippOrigin();
		applyRotation(MDaRef, rotVar, MDaRefRot); //rotation to reference image
		MultidimArray<std::complex<double> > MDaRefRotF;
		transformersForImages[id].FourierTransform(MDaRefRot, MDaRefRotF, true);
		auto &ccMatrixShift = ccMatrixShifts.at(id);
		ccMatrix(MDaInF, MDaRefRotF, ccMatrixShift, ccMatrixBestCandidTransformers.at(id)); // cross-correlation matrix

		MultidimArray<double> ccVectorTx;
		maxByColumn(ccMatrixShift, ccVectorTx); // ccvMatrix to ccVector
		double tx = 0;
		getShift(ccVectorTx, tx, XSIZE(ccMatrixShift));
		MultidimArray<double> ccVectorTy;
		maxByRow(ccMatrixShift, ccVectorTy); // ccvMatrix to ccVector
		double ty = 0;
		getShift(ccVectorTy, ty, YSIZE(ccMatrixShift));

		if (std::abs(tx) > maxShift || std::abs(ty) > maxShift)
			return;

		//apply transformation to experimental image
		double expTx;
		double expTy;
		double expPsi;
		expPsi = -rotVar;
		// applying in one transform
		MultidimArray<double> MDaInShiftRot;
		MDaInShiftRot.setXmippOrigin();
		applyShiftAndRotation(MDaIn, expPsi, tx, ty, MDaInShiftRot);

		circularWindow(MDaInShiftRot); //circular masked MDaInRotShift

		auto tempCoeff = correlationIndex(MDaRef, MDaInShiftRot);
		std::lock_guard lock(mutex);
		if (tempCoeff > bestCoeff) {
			bestCoeff = tempCoeff;
			shift_x = tx;
			shift_y = ty;
			psi = expPsi;
		}
	};

	// process peaks in parallel
	auto futures = std::vector<std::future<void>>();
	for (size_t i = 0; i < peaksFound; ++i) {
        futures.emplace_back(threadPool.push(workload, i));
    }
	// wait for processing to finish
    for (auto &f : futures) {
        f.get();
    }
}

/* apply rotation */
void ProgAngularAssignmentMag::applyRotation(const MultidimArray<double> &MDaRef, double &rot,
		MultidimArray<double> &MDaRefRot) const{
	// Transform matrix
	Matrix2D<double> A(3, 3);
	A.initIdentity();
	double ang = DEG2RAD(rot);
	double cosine = cos(ang);
	double sine = sin(ang);

	// rotation
	MAT_ELEM(A,0, 0) = cosine;
	MAT_ELEM(A,0, 1) = sine;
	MAT_ELEM(A,1, 0) = -sine;
	MAT_ELEM(A,1, 1) = cosine;

	// Shift
	MAT_ELEM(A,0, 2) = 0.;
	MAT_ELEM(A,1, 2) = 0.;

	applyGeometry(xmipp_transformation::LINEAR, MDaRefRot, MDaRef, A, xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP);
}

/* finds shift as maximum of ccVector */
void ProgAngularAssignmentMag::getShift(MultidimArray<double> &ccVector,
		double &shift, const size_t &size) const {
	double maxVal = -10.;
	int idx = 0;
	auto lb = (int)((double)size / 2. - maxShift);
	auto hb = (int)((double)size / 2. + maxShift);
	for (int i = lb; i < hb; ++i) {
		if (( dAi(ccVector,size_t(i)) > dAi(ccVector, size_t(i - 1)))
				&& ( dAi(ccVector,size_t(i)) > dAi(ccVector, size_t(i + 1))) && // is this value a peak value?
				(dAi(ccVector,i) > maxVal)) {  // is the biggest?
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
		shift = maxShift+1.;
	}

}

/* apply shift then rotation */
void ProgAngularAssignmentMag::applyShiftAndRotation(const MultidimArray<double> &MDaRef, double &rot, double &tx,
		double &ty, MultidimArray<double> &MDaRefRot) const {
	// Transform matrix
	Matrix2D<double> A(3, 3);
	A.initIdentity();
	double ang = DEG2RAD(rot);
	double cosine = cos(ang);
	double sine = sin(ang);

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

	applyGeometry(xmipp_transformation::LINEAR, MDaRefRot, MDaRef, A, xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP);
}
