/***************************************************************************
 *
 * Authors:    Amaya Jimenez    (ajimenez@cnb.csic.es)
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

#include "particle_polishing.h"

void ProgParticlePolishing::defineParams()
{
    addUsageLine("Particle polishing from a stack of movie particles");
    addParamsLine(" -i <movie>: Input movie particle metadata");
    addParamsLine(" -vol <volume>: Input volume to generate the reference projections");
    addParamsLine(" --s <samplingRate=1>: Sampling rate");
    addParamsLine(" --nFrames <nFrames>: Number of frames");
    addParamsLine(" --nMovies <nMovies>: Number of movies");
    addParamsLine(" --w <w=1>: Window size. The number of frames to average to correlate that averaged image with the projection.");
    addParamsLine(" --movxdim <xmov> : Movie size in x dimension");
    addParamsLine(" --movydim <ymov> : Movie size in y dimension");
    addParamsLine(" [-o <fnOut=\"out.xmd\">]: Output metadata with weighted particles");

}

void ProgParticlePolishing::readParams()
{
	fnPart=getParam("-i");
	fnVol = getParam("-vol");
	fnOut=getParam("-o");
	nFrames=getIntParam("--nFrames");
	nMovies=getIntParam("--nMovies");
	w=getIntParam("--w");
	samplingRate=getDoubleParam("--s");
	xmov = getIntParam("--movxdim");
	ymov = getIntParam("--movydim");

}


void ProgParticlePolishing::show()
{
	if (verbose==0)
		return;
	std::cout
	<< "Input movie particle metadata:     " << fnPart << std::endl
	<< "Input volume to generate the reference projections:     " << fnVol << std::endl
	;
}


void ProgParticlePolishing::similarity (const MultidimArray<double> &I, const MultidimArray<double> &Iexp, double &corrN, double &corrM, double &corrW, double &imed){


	MultidimArray<double> Idiff;
	Idiff=I;
	Idiff-=Iexp;
	double meanD, stdD;
	Idiff.computeAvgStdev(meanD,stdD);
	Idiff.selfABS();
	double thD=stdD;

	/*/DEBUG
	Image<double> Idiff2, I2, Iexp2;
	Idiff2() = Idiff;
	Idiff2.write("diff.mrc");
	I2()=I;
	I2.write("projection.mrc");
	Iexp2()=Iexp;
	Iexp2.write("experimental.mrc");
	//END DEBUG/*/

	double meanI, stdI, thI;
	I.computeAvgStdev(meanI,stdI);
	thI = stdI;

	//std::cerr << "- THI: " << thI << ",  THD: " << thD << std::endl;

	double NI=0;
	double ND=0;
	double sumMI=0, sumMIexp=0;
	double sumI=0, sumIexp=0;
	double sumWI=0, sumWIexp=0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Idiff)
	{
		double p=DIRECT_MULTIDIM_ELEM(I,n);
		double pexp=DIRECT_MULTIDIM_ELEM(Iexp,n);
		sumI+=p;
		sumIexp+=pexp;
		if (DIRECT_MULTIDIM_ELEM(Idiff,n)>thD)
		{
			sumWI+=p;
			sumWIexp+=pexp;
			ND+=1.0;
		}
		if(p>thI){
			sumMI+=p;
			sumMIexp+=pexp;
			NI+=1.0;
		}
	}

	double avgI, avgIexp, avgMI, avgMIexp, avgWI, avgWIexp, iNI, iND;
	double isize=1.0/MULTIDIM_SIZE(Idiff);
	avgI=sumI*isize;
	avgIexp=sumIexp*isize;
	if (NI>0){
		iNI=1.0/NI;
		avgMI=sumMI*iNI;
		avgMIexp=sumMIexp*iNI;
	}
	if(ND>0){
		iND=1.0/ND;
		avgWI=sumWI*iND;
		avgWIexp=sumWIexp*iND;
	}

	double sumIIexp=0.0, sumII=0.0, sumIexpIexp=0.0;
	double sumMIIexp=0.0, sumMII=0.0, sumMIexpIexp=0.0;
	double sumWIIexp=0.0, sumWII=0.0, sumWIexpIexp=0.0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Idiff)
	{
		double p=DIRECT_MULTIDIM_ELEM(I,n);
		double pexp=DIRECT_MULTIDIM_ELEM(Iexp,n);
		double pIa=p-avgI;
		double pIexpa=pexp-avgIexp;
		sumIIexp+=pIa*pIexpa;
		sumII +=pIa*pIa;
		sumIexpIexp +=pIexpa*pIexpa;

		if (p>thI){
			pIa=p-avgMI;
			pIexpa=pexp-avgMIexp;
			sumMIIexp+=pIa*pIexpa;
			sumMII +=pIa*pIa;
			sumMIexpIexp +=pIexpa*pIexpa;
		}
		if (DIRECT_MULTIDIM_ELEM(Idiff,n)>thD)
		{
			double w=DIRECT_MULTIDIM_ELEM(Idiff,n);
			pIa=p-avgWI;
			pIexpa=pexp-avgMIexp;
			sumWIIexp+=w*pIa*pIexpa;
			sumWII +=w*pIa*pIa;
			sumWIexpIexp +=w*pIexpa*pIexpa;
		}
	}

	sumIIexp*=isize;
	sumII*=isize;
	sumIexpIexp*=isize;

	sumMIIexp*=iNI;
	sumMII*=iNI;
	sumMIexpIexp*=iNI;

	sumWIIexp*=iND;
	sumWII*=iND;
	sumWIexpIexp*=iND;

	//corrN=sumIIexp/sqrt(sumII*sumIexpIexp);
	//corrM=sumMIIexp/sqrt(sumMII*sumMIexpIexp);
	//corrW=sumWIIexp/sqrt(sumWII*sumWIexpIexp);
	corrN=sumIIexp;
	corrM=sumMIIexp;
	corrW=sumWIIexp;
	if(std::isnan(corrN))
		corrN=-1.0;
	if(std::isnan(corrM))
		corrM=-1.0;
	if(std::isnan(corrW))
		corrW=-1.0;
	imed=imedDistance(I, Iexp);

}

void ProgParticlePolishing::averagingMovieParticles(MetaData &mdPart, MultidimArray<double> &I, size_t partId, size_t frameId, size_t movieId, int window){


	MDRow currentRow;
	FileName fnPart;
	Image<double> Ipart;
	size_t newPartId, newFrameId, newMovieId;
	size_t mdPartSize = mdPart.size();
	if(window%2==1)
		window+=1;
	int w=window/2;
	double count=1.0;
	MDIterator *iterPart = new MDIterator(mdPart);
	int frameIdI = int(frameId);
	for(int i=0; i<mdPartSize; i++){

		mdPart.getRow(currentRow, iterPart->objId);
		currentRow.getValue(MDL_PARTICLE_ID, newPartId);
		currentRow.getValue(MDL_MICROGRAPH_ID, newMovieId);
		if((newPartId==partId) && (newMovieId==movieId)){
			currentRow.getValue(MDL_FRAME_ID, newFrameId);
			int newFrameIdI = int(newFrameId);
			if(newFrameIdI==frameIdI){
				if(iterPart->hasNext())
					iterPart->moveNext();
				continue;
			}
			if (newFrameIdI<frameIdI){ //taking into account all the previous frames
				//std::cerr << count << ". Encuentra frames para el averaging: " << partId << " , " << newPartId << " , " << frameIdI << " , " << newFrameIdI << " , " << movieId << " , " << newMovieId << std::endl;
				currentRow.getValue(MDL_IMAGE,fnPart);
				Ipart.read(fnPart);
				Ipart().setXmippOrigin();
				I+=Ipart();
				count+=1.0;
			}
			else{
				if(iterPart->hasNext())
					iterPart->moveNext();
				continue;
			}
		}
		else{
			if(iterPart->hasNext())
				iterPart->moveNext();
			continue;
		}

		if(iterPart->hasNext())
			iterPart->moveNext();

	} //end loop to find the previous and following frames to average the particle

	//I/=count;

}



void ProgParticlePolishing::calculateFrameWeightPerFreq(MultidimArray<double> &matrixWeights, MultidimArray<double> &weightsperfreq, const MultidimArray<double> &maxvalues){

	std::cerr << "CALCULATING frame weights per freq..." << std::endl;
    for (size_t l=0; l<NSIZE(matrixWeights); ++l){ //loop over movies
        for (size_t k=0; k<ZSIZE(matrixWeights); ++k){ //loop over frames
            for (size_t i=0; i<YSIZE(matrixWeights); ++i){ //loop over frequencies
                for (size_t j=0; j<XSIZE(matrixWeights); ++j){ //loop over movie particles
                	if(DIRECT_NZYX_ELEM(matrixWeights, l, k, i, j)==0)
                		continue;
                	DIRECT_NZYX_ELEM(weightsperfreq, l, k, i, j) = (DIRECT_NZYX_ELEM(matrixWeights, l, k, i, j)/DIRECT_A1D_ELEM(maxvalues,l));
                	std::cerr << "- Movie: " << l+1 << ". Frame: " << k+1 << ". Freq: " << i << ". Valores: " << DIRECT_NZYX_ELEM(weightsperfreq, l, k, i, j) << " , " << DIRECT_NZYX_ELEM(matrixWeights, l, k, i, j) << " , " << DIRECT_A1D_ELEM(maxvalues,l) << std::endl;
                }
            }
        }
    }

}


void ProgParticlePolishing::smoothingWeights(MultidimArray<double> &in, MultidimArray<double> &out){

	std::cerr << "SMOOTHING values..." << std::endl;
	MultidimArray<double> aux(ZSIZE(in), YSIZE(in));
	MultidimArray<double> coeffs(ZSIZE(in), YSIZE(in));

    for (size_t l=0; l<NSIZE(in); ++l){ //loop over movies

    	for (size_t j=0; j<XSIZE(in); ++j){ //loop over movie particles
    		for (size_t k=0; k<ZSIZE(in); ++k){ //loop over frames
    			for (size_t i=0; i<YSIZE(in); ++i){ //loop over frequencies
    				DIRECT_A2D_ELEM(aux, k, i)=DIRECT_NZYX_ELEM(in, l, k, i, j);
                }
            }

        }

        //TODO: change this smoothing by low rank tensor decomposition
    	produceSplineCoefficients(BSPLINE3,coeffs,aux);

        for (size_t j=0; j<XSIZE(in); ++j){ //loop over movie particles
        	for (size_t k=0; k<ZSIZE(in); ++k){ //loop over frames
        		for (size_t i=0; i<YSIZE(in); ++i){ //loop over frequencies
                	DIRECT_NZYX_ELEM(out, l, k, i, j)=DIRECT_A2D_ELEM(coeffs, k, i);
                }
            }
        }

    }
}


void ProgParticlePolishing::produceSideInfo()
{

	int a=0;

	/*
	//DEBUGING the correct way of applyGeometry
	size_t Xdim, Ydim, Zdim, Ndim;
	MetaData mdPart, mdRef;
	mdPart.read("images_iter001_00.xmd");
	size_t mdPartSize = mdPart.size();

	MDIterator *iterPart = new MDIterator();
	MDIterator *iterRef = new MDIterator();
	FileName fnPart, fnRef;
	Image<double> Ipart, Iref;
	MDRow currentRow;
	Matrix2D<double> A;
	Projection PV;

	V.read("axes.vol");
    V().setXmippOrigin();
	projectorV = new FourierProjector(V(),2,0.5,BSPLINE3);
	int xdim = (int)XSIZE(V());

	iterPart->init(mdPart);

	double rot, tilt, psi, x, y;
	for(int i=0; i<mdPartSize; i++){
		//Project the volume with the parameters in the image

		bool flip;
		size_t frId, mvId, partId;
		mdPart.getRow(currentRow, iterPart->objId);
		currentRow.getValue(MDL_IMAGE,fnPart);
		Ipart.read(fnPart);
		Ipart().setXmippOrigin();
		currentRow.getValue(MDL_ANGLE_ROT,rot);
		currentRow.getValue(MDL_ANGLE_TILT,tilt);
		currentRow.getValue(MDL_ANGLE_PSI,psi);
		currentRow.getValue(MDL_SHIFT_X,x);
		currentRow.getValue(MDL_SHIFT_Y,y);
		currentRow.getValue(MDL_FLIP,flip);
		currentRow.getValue(MDL_FRAME_ID,frId);
		currentRow.getValue(MDL_MICROGRAPH_ID,mvId);
		currentRow.getValue(MDL_PARTICLE_ID,partId);

		A.initIdentity(3);
		MAT_ELEM(A,0,2)=x;
		MAT_ELEM(A,1,2)=y;
		if (flip)
		{
			MAT_ELEM(A,0,0)*=-1;
			MAT_ELEM(A,0,1)*=-1;
			MAT_ELEM(A,0,2)*=-1;
		}
	}

	std::cerr << fnRef << ", " << fnPart << ", " << x << ", " << y << ", " << psi << ", " << rot << ", " << tilt << ", " << A << std::endl;
	Image<double> Iout;
	projectVolume(*projectorV, PV, xdim, xdim, rot, tilt, psi);
	applyGeometry(LINEAR,Iout(),PV(),A,IS_INV,DONT_WRAP,0.);
	Iout().setXmippOrigin();
	Ipart.write("Ipart.mrc");
	Iout.write("Iout2.mrc");
	exit(0);
	*/


}

void ProgParticlePolishing::averagingAll(const MetaData &mdPart, const MultidimArray<double> &I, MultidimArray<double> &Iout, size_t partId, size_t frameId, size_t movieId, bool noCurrent){

	MDRow currentRow;
	FileName fnPart;
	Image<double> Ipart;
	size_t newPartId, newFrameId, newMovieId;
	size_t mdPartSize = mdPart.size();
	double count;

	//To average the movie particle with all the frames
	Iout.initZeros(I);
	if (noCurrent){
		count=0.0;
	}else{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
				DIRECT_MULTIDIM_ELEM(Iout,n)=DIRECT_MULTIDIM_ELEM(I,n);
		count=1.0;
	}
	MDIterator *iterPart = new MDIterator(mdPart);
	int frameIdI = int(frameId);
	for(int i=0; i<mdPartSize; i++){

		mdPart.getRow(currentRow, iterPart->objId);
		currentRow.getValue(MDL_PARTICLE_ID, newPartId);
		currentRow.getValue(MDL_MICROGRAPH_ID, newMovieId);
		if((newPartId==partId) && (newMovieId==movieId)){
			currentRow.getValue(MDL_FRAME_ID, newFrameId);
			int newFrameIdI = int(newFrameId);
			if(newFrameIdI==frameIdI){
				if(iterPart->hasNext())
					iterPart->moveNext();
				continue;
			}
			else{ //averaging with all the frames
				currentRow.getValue(MDL_IMAGE,fnPart);
				Ipart.read(fnPart);
				Ipart().setXmippOrigin();
				Iout+=Ipart();
				count+=1.0;
			}
		}
		else{
			if(iterPart->hasNext())
				iterPart->moveNext();
			continue;
		}

		if(iterPart->hasNext())
			iterPart->moveNext();

	} //end loop to find the previous and following frames to average the particle
	Iout/=count;

}



void ProgParticlePolishing::calculateCurve(const MultidimArray<double> &Iavg, const MultidimArray<double> &Iproj, double &slope, double &intercept){

	int nStep=30;
	MultidimArray<double> vectorAvg;
	vectorAvg.initZeros(2, nStep);
	double Dmin, Dmax;
	Iproj.computeDoubleMinMax(Dmin, Dmax);
	double step = (Dmax-Dmin)/double(nStep);
	double offset = -Dmin/double(step);
	//std::cerr << "variables: " << step << ", " << offset << ", " << Dmin << ", " << Dmax <<std::endl;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Iproj){
		int pos = int(floor((DIRECT_MULTIDIM_ELEM(Iproj, n)/step)+offset));
		if(pos>=nStep)
			pos=nStep-1;
		DIRECT_A2D_ELEM(vectorAvg, 0, pos)+=1;
		DIRECT_A2D_ELEM(vectorAvg, 1, pos)+=DIRECT_MULTIDIM_ELEM(Iavg, n);
		//if (n==20544){
		//if (pos==0){
		//	std::cerr << n << ", " << pos << ", " << DIRECT_MULTIDIM_ELEM(Iproj, n) << ", " << DIRECT_MULTIDIM_ELEM(Iavg, n) << ", " << DIRECT_A2D_ELEM(vectorAvg, 1, pos) << ", " << DIRECT_A2D_ELEM(vectorAvg, 0, pos) << std::endl;
		//}
	}
	//std::cerr << "vectorAvg calculado: " << vectorAvg << std::endl;
	std::vector<double> x, y;
	for(int n=0; n<nStep; n++){
		if (DIRECT_A2D_ELEM(vectorAvg, 0, n)!=0)
			DIRECT_A2D_ELEM(vectorAvg, 1, n)/=DIRECT_A2D_ELEM(vectorAvg, 0, n);
		x.push_back(double(n));
		y.push_back(DIRECT_A2D_ELEM(vectorAvg, 1, n));
	}

	double xSum=0, ySum=0, xxSum=0, xySum=0;
	double n = y.size();
	for (int i = 0; i < n; i++){
		xSum += x[i];
		ySum += y[i];
		xxSum += x[i] * x[i];
		xySum += x[i] * y[i];
	}
	slope = (n * xySum - xSum * ySum) / (n * xxSum - xSum * xSum);
	intercept = (ySum - slope * xSum) / n;
}


void ProgParticlePolishing::calculateBSplineCoeffs(MultidimArray<double> &inputMat, int boxsize, Matrix1D<double> &cij, int xdim, int ydim, int dataRow)
{

	//int Nx = xdim/boxsize;
	//int Ny = ydim/boxsize;

	size_t Nelements = XSIZE(inputMat);

	// For the spline regression
	//int lX=std::min(8,Nx-2), lY=std::min(8,Ny-2);
	int lX=8, lY=8;
    WeightedLeastSquaresHelper helper;
    /*helper.A.initZeros(Nx*Ny,lX*lY);
    helper.b.initZeros(Nx*Ny);
    helper.w.initZeros(Nx*Ny);
    helper.w.initConstant(1);*/
    helper.A.initZeros(Nelements, lX*lY);
    helper.b.initZeros(Nelements);
    helper.w.initZeros(Nelements);
    helper.w.initConstant(1);
    double hX = xdim / (double)(lX-3);
    double hY = ydim / (double)(lY-3);

	if ( (xdim<boxsize) || (ydim<boxsize) )
		std::cout << "Error: The input matrix to the BSliple coeffs estimation in x-direction or y-direction is too small" << std::endl;

	//std::cout << "1 Checking stuff: " << Nelements << std::endl;
    for (int i=0; i<Nelements; i++){
    	VEC_ELEM(helper.b,i)=DIRECT_A2D_ELEM(inputMat, dataRow, i);
    	int x, y;
    	x = DIRECT_A2D_ELEM(inputMat, 0, i);
    	y = DIRECT_A2D_ELEM(inputMat, 1, i);
        double xarg = x / hX;
        double yarg = y / hY;
        //std::cout << "2 Checking stuff: " << x << ", " << y << ", " << xarg << ", " << yarg << ", " << std::endl;
        for (int m = -1; m < (lY-1); m++){
            for (int l = -1; l < (lX-1); l++){
                double coeff=0.;
                coeff = Bspline03(xarg - l) * Bspline03(yarg - m);
                MAT_ELEM(helper.A,i,(m+1)*lX+(l+1)) = coeff;
            }
        }

    }
	// Spline coefficients
	weightedLeastSquares(helper, cij);
}


void ProgParticlePolishing::evaluateBSpline(const MultidimArray<double> inputMat, const Matrix1D<double> cij, MultidimArray<double> &outputMat, int xdim, int ydim, int dataRow)
{

	//int Nx = xdim/boxsize;
	//int Ny = ydim/boxsize;

	size_t Nelements = XSIZE(inputMat);

	// For the spline evaluation
	//int lX=std::min(8,Nx-2), lY=std::min(8,Ny-2);
	int lX=8, lY=8;
    double hX = xdim / (double)(lX-3);
    double hY = ydim / (double)(lY-3);

    for (int i=0; i<Nelements; i++){
    	int x, y;
    	x = DIRECT_A2D_ELEM(inputMat, 0, i);
    	y = DIRECT_A2D_ELEM(inputMat, 1, i);
        double xarg = x / hX;
        double yarg = y / hY;
        //std::cout << "2 Checking stuff: " << x << ", " << y << ", " << xarg << ", " << yarg << ", " << std::endl;
        for (int m = -1; m < (lY-1); m++){
        	double tmpY = Bspline03(yarg - m);
        	if (tmpY == 0.0)
        		continue;
        	double xContrib=0.0;
            for (int l = -1; l < (lX-1); l++){
            	double tmpX = Bspline03(xarg - l);
            	xContrib+=VEC_ELEM(cij,(m+1)*lX+(l+1)) * tmpX;
            }
            DIRECT_A2D_ELEM(outputMat, dataRow, i)+=xContrib*tmpY;
        }

    }

}


void ProgParticlePolishing::run()
{
	produceSideInfo();

	//MOVIE PARTICLES IMAGES
	MetaData mdPart;
	size_t Xdim, Ydim, Zdim, Ndim;
	mdPart.read(fnPart,NULL);
	size_t mdPartSize = mdPart.size();

	MDIterator *iterPart = new MDIterator();
	FileName fnPart;
	Image<double> Ipart, projV, Iavg;
	MDRow currentRow;
	Matrix2D<double> A;
	MultidimArray<double> dataArray, softArray;
	Projection PV;
	CTFDescription ctf;

	//INPUT VOLUME
	V.read(fnVol);
    V().setXmippOrigin();
	int xdim = (int)XSIZE(V());
	int ydim = (int)YSIZE(V());
	projectorV = new FourierProjector(V(),2,0.5,BSPLINE3);

	FourierFilter Filter;
	Filter.FilterBand=LOWPASS;
	Filter.FilterShape=RAISED_COSINE;


	/////////////////////////
	//First Part
	/////////////////////////
	std::vector<int> partIdDone;
	std::vector<int>::iterator it;
	int dataInMovie;

	for(int m=0; m<nMovies; m++){

		std::vector<int> xcoords;
		std::vector<int> ycoords;
		std::vector<double> slopes;
		std::vector<double> intercepts;
		iterPart->init(mdPart);
		dataInMovie = 0;

		for(int i=0; i<mdPartSize; i++){

			//Project the volume with the parameters in the image
			double rot, tilt, psi, x, y;
			bool flip;
			size_t frId, mvId, partId;
			int xcoor, ycoor;

			mdPart.getRow(currentRow, iterPart->objId);
			currentRow.getValue(MDL_IMAGE,fnPart);
			Ipart.read(fnPart);
			Ipart().setXmippOrigin();
			currentRow.getValue(MDL_ANGLE_ROT,rot);
			currentRow.getValue(MDL_ANGLE_TILT,tilt);
			currentRow.getValue(MDL_ANGLE_PSI,psi);
			currentRow.getValue(MDL_SHIFT_X,x);
			currentRow.getValue(MDL_SHIFT_Y,y);
			currentRow.getValue(MDL_FLIP,flip);
			currentRow.getValue(MDL_FRAME_ID,frId);
			currentRow.getValue(MDL_MICROGRAPH_ID,mvId);
			currentRow.getValue(MDL_PARTICLE_ID,partId);
			currentRow.getValue(MDL_XCOOR,xcoor);
			currentRow.getValue(MDL_YCOOR,ycoor);

			if(mvId!=m){
				if(iterPart->hasNext())
					iterPart->moveNext();
				continue;
			}

			it = find(partIdDone.begin(), partIdDone.end(), (int)partId);
			if (it != partIdDone.end()){
				if(iterPart->hasNext())
					iterPart->moveNext();
				continue;
			}
			partIdDone.push_back((int)partId);
			dataInMovie++;

			A.initIdentity(3);
			MAT_ELEM(A,0,2)=x;
			MAT_ELEM(A,1,2)=y;
			if (flip)
			{
				MAT_ELEM(A,0,0)*=-1;
				MAT_ELEM(A,0,1)*=-1;
				MAT_ELEM(A,0,2)*=-1;
			}

			if ((xdim != XSIZE(Ipart())) || (ydim != YSIZE(Ipart())))
				std::cout << "Error: The input particles and volume have different sizes" << std::endl;

			projectVolume(*projectorV, PV, xdim, xdim,  rot, tilt, psi);
			applyGeometry(LINEAR,projV(),PV(),A,IS_INV,DONT_WRAP,0.);
			projV().setXmippOrigin();

			//to obtain the points of the curve (intensity in the projection) vs (counted electrons)
			//the movie particles are averaged (all frames) to compare every pixel value
			averagingAll(mdPart, Ipart(), Iavg(), partId, frId, mvId, false);

			//With Iavg and projV, we calculate the curve (intensity in the projection) vs (counted electrons)
			double slope=0, intercept=0;
			calculateCurve(Iavg(), projV(), slope, intercept);
			std::cerr << ". Movie: " << mvId << ". Frame: " << frId << ". ParticleId: " << partId << ". Slope: " << slope << ". Intercept: " << intercept <<  std::endl;

			//Vectors to store some results
			xcoords.push_back((int)xcoor);
			ycoords.push_back((int)ycoor);
			slopes.push_back(slope);
			intercepts.push_back(intercept);

			/*//DEBUG
			projV.write(formatString("Testprojection_%i_%i.tif", frId, partId));
			//Ipart.write(formatString("particle_%i_%i.tif", frId, partId));
			Iavg.write(formatString("Testaverage_%i_%i.tif", frId, partId));
			//END DEBUG/*/

			if(iterPart->hasNext())
				iterPart->moveNext();

		}//end particles loop

		if(dataInMovie>0){
			Matrix1D<double> cij_slopes, cij_intercepts;
			int boxsize = 50;
			dataArray.initZeros(4, dataInMovie);
			softArray.initZeros(4, dataInMovie);
			for(int h=0; h<dataInMovie; h++){
				//Data array will be used to store some results
				//Data array with xcoor in the first column
				DIRECT_A2D_ELEM(dataArray, 0, h) = xcoords[h];
				DIRECT_A2D_ELEM(softArray, 0, h) = xcoords[h];
				//Data array with ycoor in the second column
				DIRECT_A2D_ELEM(dataArray, 1, h) = ycoords[h];
				DIRECT_A2D_ELEM(softArray, 1, h) = ycoords[h];
				//Data array with ycoor in the second column
				DIRECT_A2D_ELEM(dataArray, 2, h) = slopes[h];
				//Data array with ycoor in the second column
				DIRECT_A2D_ELEM(dataArray, 3, h) = intercepts[h];
			}

			/*/AJ testing
			dataArray.resize(4, floor(xmov/1000)*floor(ymov/1000));
			std::mt19937_64 generator;
			generator.seed(std::time(0));
			std::normal_distribution<double> normal;
			int count=0;
			for(int h=0; h<floor(xmov/1000); h++){
				for(int l=0; l<floor(ymov/1000); l++){
					DIRECT_A2D_ELEM(dataArray, 0, count) = h;
					DIRECT_A2D_ELEM(dataArray, 1, count) = l;
					DIRECT_A2D_ELEM(dataArray, 2, count) = -1*normal(generator);
					DIRECT_A2D_ELEM(dataArray, 3, count) = normal(generator);
					count++;
				}
			}
			//std::cout << "dataArray " << dataArray << std::endl;
			//END AJ testing*/

			//AJ to obtain a soft version of the obtained slopes (dataRow=2)
			std::cout << dataInMovie << " Slopes and intercepts " << dataArray << std::endl;

			calculateBSplineCoeffs(dataArray, boxsize, cij_slopes, xmov, ymov, 2);
			std::cout << dataInMovie << " Slopes coeffs " << cij_slopes << std::endl;
			evaluateBSpline(dataArray, cij_slopes, softArray, xmov, ymov, 2);

			//AJ to obtain a soft version of the obtained interception points (dataRow=3)
			calculateBSplineCoeffs(dataArray, boxsize, cij_intercepts, xmov, ymov, 3);
			std::cout << dataInMovie << " Intercepts coeffs " << cij_intercepts << std::endl;
			evaluateBSpline(dataArray, cij_intercepts, softArray, xmov, ymov, 3);

			std::cout << dataInMovie << " SOFT Slopes and intercepts " << softArray << std::endl;

		}

	}//end movie Ids loop

	/////////////////////////
	//Second Part
	/////////////////////////
	MultidimArray<double> matrixWeights, maxvalues;
	double cutfreq;
	double inifreq=0.05; //0.05;
	double step=0.05; //0.05;
	int Nsteps= int((0.5-inifreq)/step);
	double maxvalue=-1.;
	size_t mvId, frId, partId;


	iterPart->init(mdPart);
	std::vector<int> numPartPerMov;
	int prevMvId=-1, prevPartId=-1, partCount=0;
	for(int i=0; i<mdPartSize; i++){
		size_t mvId, partId;
		mdPart.getRow(currentRow, iterPart->objId);
		currentRow.getValue(MDL_MICROGRAPH_ID,mvId);
		currentRow.getValue(MDL_PARTICLE_ID,partId);
		if(mvId!=prevMvId){
			if (prevMvId!=-1)
				numPartPerMov.push_back(partCount);
			partCount=0;
			prevMvId=mvId;
		}
		if(partId!=prevPartId){
			prevPartId=partId;
			partCount++;
		}
		if(iterPart->hasNext())
			iterPart->moveNext();
	}
	numPartPerMov.push_back(partCount);
	std::cerr << " size numPartPerMov " << numPartPerMov.size() << std::endl;
	for (int i=0; i<numPartPerMov.size(); i++)
		std::cerr << " numPartPerMov[" << i << "] = "<< numPartPerMov[i] << std::endl;

	prevMvId=-1;
	prevPartId=-1;
	partCount=0;
	int countMv=0;
	iterPart->init(mdPart);
	Image<double> Ipartaux, projVaux;
	for(int i=0; i<mdPartSize; i++){
		
		//Project the volume with the parameters in the image
		double rot, tilt, psi, x, y;
		bool flip;
		size_t frId, mvId, partId;
		int xcoor, ycoor;

		mdPart.getRow(currentRow, iterPart->objId);
		currentRow.getValue(MDL_IMAGE,fnPart);
		Ipart.read(fnPart);
		Ipart().setXmippOrigin();
		currentRow.getValue(MDL_ANGLE_ROT,rot);
		currentRow.getValue(MDL_ANGLE_TILT,tilt);
		currentRow.getValue(MDL_ANGLE_PSI,psi);
		currentRow.getValue(MDL_SHIFT_X,x);
		currentRow.getValue(MDL_SHIFT_Y,y);
		currentRow.getValue(MDL_FLIP,flip);
		currentRow.getValue(MDL_FRAME_ID,frId);
		currentRow.getValue(MDL_MICROGRAPH_ID,mvId);
		currentRow.getValue(MDL_PARTICLE_ID,partId);
		currentRow.getValue(MDL_XCOOR,xcoor);
		currentRow.getValue(MDL_YCOOR,ycoor);

		if(prevMvId!=mvId){
			if (prevMvId!=-1){
				std::cerr << " maxvalue " << maxvalue << std::endl;
				std::cerr << "  matrixWeights " << matrixWeights << std::endl;
				matrixWeights/=maxvalue;
				std::cerr << " NORM matrixWeights " << matrixWeights << std::endl;
				countMv++;
			}
			partCount=0;
			prevMvId=mvId;
			maxvalue=-1.;
			matrixWeights.initZeros(numPartPerMov[countMv], nFrames, Nsteps+1);
		}
		if(partId!=prevPartId){
			prevPartId=partId;
			partCount++;
		}

		A.initIdentity(3);
		MAT_ELEM(A,0,2)=x;
		MAT_ELEM(A,1,2)=y;
		if (flip){
			MAT_ELEM(A,0,0)*=-1;
			MAT_ELEM(A,0,1)*=-1;
			MAT_ELEM(A,0,2)*=-1;
		}

		projectVolume(*projectorV, PV, xdim, xdim,  rot, tilt, psi);
		applyGeometry(LINEAR,projV(),PV(),A,IS_INV,DONT_WRAP,0.);
		projV().setXmippOrigin();
		//TODO: ver si estamos alineando en sentido correcto la proyeccion - hecho en el debugging en el produceSideInfo

		//TODO: invertir contraste y aplicar ctf, o al reves, no se...
		//To invert contrast in the projections
		double Dmin, Dmax, irange, val;
		projV().computeDoubleMinMax(Dmin, Dmax);
		irange=1.0/(Dmax - Dmin);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(projV()){
			val=DIRECT_MULTIDIM_ELEM(projV(),n);
			DIRECT_MULTIDIM_ELEM(projV(),n) = (Dmax - val) * irange;
		}

		//filtering the projections with the ctf
		ctf.readFromMdRow(currentRow);
		ctf.produceSideInfo();
		ctf.applyCTF(projV(), samplingRate, false);

		//averaging movie particle image with the ones in all the frames but without the current one
		//averagingMovieParticles(mdPart, Ipart(), partId, frId, mvId, w);
		averagingAll(mdPart, Ipart(), Iavg(), partId, frId, mvId, true);
		//TODO check how we have to average the movie particles to calculate the correlations

		for(int n=0; n<Nsteps+1; n++){

			if(n==0){
				projVaux().resize(projV());
				Ipartaux().resize(Ipart());
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(projV()){
						DIRECT_MULTIDIM_ELEM(projVaux(),n) = DIRECT_MULTIDIM_ELEM(projV(),n);
				}
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Ipart()){
						DIRECT_MULTIDIM_ELEM(Ipartaux(),n) = DIRECT_MULTIDIM_ELEM(Ipart(),n);
				}
				projVaux().setXmippOrigin();
				Ipartaux().setXmippOrigin();
			}else{
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(projV()){
						DIRECT_MULTIDIM_ELEM(projV(),n) = DIRECT_MULTIDIM_ELEM(projVaux(),n);
				}
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Ipart()){
						DIRECT_MULTIDIM_ELEM(Ipart(),n) = DIRECT_MULTIDIM_ELEM(Ipartaux(),n);
				}
			}
			//TODO: a partir de aqui trabajar con las projVaux y Ipartaux

			cutfreq = inifreq + step*n;

			//filtering the projected particles with the lowpass filter
			Filter.w1=cutfreq;
			Filter.generateMask(projV());
			Filter.applyMaskSpace(projV());

			//filtering the averaged movie particle
			Filter.generateMask(Ipart());
			Filter.applyMaskSpace(Ipart());

			//calculate similarity measures between averaged movie particles and filtered projection
			double weight;
			double corrN, corrM, corrW, imed;
			similarity(projV(), Ipart(), corrN, corrM, corrW, imed);
			//TODO: la corrW que viene de Idiff no tiene mucho sentido porque la Idiff en este caso no nos dice nada

			weight = corrN; //TODO
			if(weight>maxvalue)
				maxvalue=weight;

			//std::cerr << "- Freq: " << cutfreq << ". Movie: " << mvId << ". Frame: " << frId << ". ParticleId: " << partId << ". CorrN: " << corrN << ". CorrM: " << corrM << ". CorrW: " << corrW << ". Imed: " << imed << std::endl;


			//To align averaged movie particle and projection
			//Matrix2D<double> M;
			//MultidimArray<double> IpartAlign(Ipart());
			//IpartAlign = Ipart();
			//alignImages(projV(), IpartAlign, M, false);
			//MultidimArray<double> shiftXmatrix;
			//shiftXmatrix.initZeros(matrixWeights);
			//MultidimArray<double> shiftYmatrix;
			//shiftYmatrix.initZeros(matrixWeights);
			//if(MAT_ELEM(M,0,2)<10 && MAT_ELEM(M,1,2)<10){
			//	DIRECT_NZYX_ELEM(shiftXmatrix, mvId-1, frId-1, n, i) = MAT_ELEM(M,0,2);
			//	DIRECT_NZYX_ELEM(shiftYmatrix, mvId-1, frId-1, n, i) = MAT_ELEM(M,1,2);
			//}
			//std::cerr << "- Transformation matrix: " << M  << std::endl;
			//std::cerr << MAT_ELEM(M,0,2) << " " << MAT_ELEM(M,1,2) << std::endl;

			DIRECT_ZYX_ELEM(matrixWeights, partCount-1, frId-1, n) = weight;

		} //end frequencies loop


		/*/DEBUG
		if(frId==nFrames){
			projV.write(formatString("projection_%i_%i.tif", frId, partId));
			//Ipart.write(formatString("particle_%i_%i.tif", frId, partId));
			Iavg.write(formatString("average_%i_%i.tif", frId, partId));
		}
		//END DEBUG/*/

		if(iterPart->hasNext())
			iterPart->moveNext();

	} //end movie particles loop




	//MultidimArray<double> weightsperfreq;
	//weightsperfreq.initZeros(NSIZE(matrixWeights), ZSIZE(matrixWeights), YSIZE(matrixWeights), XSIZE(matrixWeights));
	//calculateFrameWeightPerFreq(matrixWeights, weightsperfreq, maxvalues);

}



