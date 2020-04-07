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

#include <iostream>
#include <random>
#include "particle_polishing.h"
#include <iterator>

#define sqr(x) x*x

void ProgParticlePolishing::defineParams()
{
    addUsageLine("Particle polishing from a stack of movie particles");
    addParamsLine(" -i <movie>: Input movie particle metadata");
    //addParamsLine(" -iPart <part>: Input particles metadata");
    addParamsLine(" -vol <volume>: Input volume to generate the reference projections");
    addParamsLine(" --s <samplingRate=1>: Sampling rate");
    addParamsLine(" --nFrames <nFrames>: Number of frames");
    addParamsLine(" --nMics <nMics>: Number of micrographs");
    addParamsLine(" --w <w=1>: Window size. The number of frames to average to correlate that averaged image with the projection.");
    addParamsLine(" --movxdim <xmov> : Movie size in x dimension");
    addParamsLine(" --movydim <ymov> : Movie size in y dimension");
    addParamsLine(" [-o <fnOut=\"out.xmd\">]: Output metadata with weighted particles");

}

void ProgParticlePolishing::readParams()
{
	fnMdMov=getParam("-i");
	//fnMdPart=getParam("-iPart");
	fnVol = getParam("-vol");
	fnOut=getParam("-o");
	nFrames=getIntParam("--nFrames");
	nMics=getIntParam("--nMics");
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
	<< "Input movie particle metadata:     " << fnMdMov << std::endl
	<< "Input volume to generate the reference projections:     " << fnVol << std::endl
	;
}


void ProgParticlePolishing::similarity (const MultidimArray<double> &I, const MultidimArray<double> &Iexp, double &corrN,
		double &corrM, double &corrW, double &imed, const double &meanF=0.){


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
	if(meanF!=0.){
		//printf("Changing the mean of the experimental images %lf %lf \n", avgIexp, meanF);
		avgIexp=meanF;
		//avgI=100;
	}//else{
		//printf("NO changing the mean of the experimental images %lf \n", avgIexp);
	//}

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
	//for(int n=89980; n<90000; n++)
	{
		double p=DIRECT_MULTIDIM_ELEM(I,n);
		double pexp=DIRECT_MULTIDIM_ELEM(Iexp,n);
		double pIa=p-avgI;
		double pIexpa=pexp-avgIexp;
		sumIIexp+=(pIa*pIexpa);
		sumII +=(pIa*pIa);
		sumIexpIexp +=(pIexpa*pIexpa);
		//if (n<20 || (n>44990 && n<45010) || n>89980)
			//printf("%d Debug %lf %lf %lf %lf %lf %lf %lf \n", n, pexp, avgIexp, pIexpa, p, avgI, pIa, sumIIexp);

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

	//printf("Some values %lf, %lf, ", sumIIexp, isize);
	sumIIexp*=isize;
	//printf(" %lf, ", sumIIexp);
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
	//printf(" %lf \n", corrN);
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



void ProgParticlePolishing::calculateCurve_1(const MultidimArray<double> &Iavg, const MultidimArray<double> &Iproj, MultidimArray<double> &vectorAvg, int nStep, double step, double offset, double Dmin, double Dmax){

	//int nStep=30;
	//MultidimArray<double> vectorAvg;
	//vectorAvg.initZeros(2, nStep);
	//double Dmin, Dmax;
	//Iproj.computeDoubleMinMax(Dmin, Dmax);
	//double step = (Dmax-Dmin)/double(nStep);
	//double offset = -Dmin/double(step);
	//std::cerr << "variables: " << step << ", " << offset << ", " << Dmin << ", " << Dmax <<std::endl;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Iproj){
		if(DIRECT_MULTIDIM_ELEM(Iproj, n)<Dmin)
			DIRECT_MULTIDIM_ELEM(Iproj, n)=Dmin;
		if(DIRECT_MULTIDIM_ELEM(Iproj, n)>Dmax)
			DIRECT_MULTIDIM_ELEM(Iproj, n)=Dmax;
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

}


void ProgParticlePolishing::calculateCurve_2(const MultidimArray<double> &Iproj, MultidimArray<double> &vectorAvg, int nStep, double &slope, double &intercept, double Dmin, double Dmax){

	//std::cerr << "vectorAvg calculado: " << vectorAvg << std::endl;
	//double Dmin, Dmax;
	//Iproj.computeDoubleMinMax(Dmin, Dmax);
	std::vector<double> x, y;
	double aux;
	for(int n=0; n<nStep; n++){
		if (DIRECT_A2D_ELEM(vectorAvg, 0, n)!=0){
			DIRECT_A2D_ELEM(vectorAvg, 1, n)/=DIRECT_A2D_ELEM(vectorAvg, 0, n);
			aux = Dmin + n*((Dmax-Dmin)/(nStep-1));
			x.push_back(double(aux));
			y.push_back(DIRECT_A2D_ELEM(vectorAvg, 1, n));
		}
	}

	/*std::cout << "Vector X " << std::endl;
	for (std::vector<double>::const_iterator it = x.begin(); it != x.end(); ++it)
	{
	    std::cout << *it << std::endl;
	}
	std::cout << "Vector Y " << std::endl;
	for (std::vector<double>::const_iterator it = y.begin(); it != y.end(); ++it)
	{
	    std::cout << *it << std::endl;
	}*/

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
	MetaData mdPartPrev, mdPart;
	size_t Xdim, Ydim, Zdim, Ndim;
	mdPartPrev.read(fnMdMov,NULL);
	size_t mdPartSize = mdPartPrev.size();

	mdPart.sort(mdPartPrev, MDL_PARTICLE_ID);

	MDIterator *iterPart = new MDIterator();
	MDIterator *iterPart2 = new MDIterator();
	FileName fnPart;
	Image<double> Ipart, projV, Iavg, Iproj, IpartOut, Iout, Ifinal;
	MDRow currentRow;
	MDRow currentRow2;
	Matrix2D<double> A, Aout;
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
	std::vector<int> mvIds;
	std::vector<double> slopes;
	std::vector<double> intercepts;
	double slope=0., intercept=0.;
	int nStep=30;
	MultidimArray<double> vectorAvg;

	double Dmin=0., Dmax=0.;
	double stepCurve;
	double offset;


	for(int m=0; m<nMics; m++){

		double rot, tilt, psi, x, y;
		bool flip;
		size_t frId, mvId, partId;
		int xcoor, ycoor;
		int enabled;

		iterPart->init(mdPart);
		dataInMovie = 0;
		vectorAvg.initZeros(2, nStep);
		slope=0.;
		intercept=0.;

		for(int i=0; i<mdPartSize; i++){

			//Project the volume with the parameters in the image
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
			currentRow.getValue(MDL_ENABLED,enabled);

			//std::cout << mvId << " " << frId << " " << partId << std::endl;
			if(enabled==-1){
				partIdDone.push_back((int)partId);
				if(iterPart->hasNext())
					iterPart->moveNext();
				continue;
			}

			if(mvId>m+1){
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
			//AJ TODO: falta incluir la CTF?

			//to obtain the points of the curve (intensity in the projection) vs (counted electrons)
			//the movie particles are averaged (all frames) to compare every pixel value
			averagingAll(mdPart, Ipart(), Iavg(), partId, frId, mvId, false);

			//With Iavg and projV, we calculate the curve (intensity in the projection) vs (counted electrons)
			if(Dmin==0. && Dmax==0.){
				projV().computeDoubleMinMax(Dmin, Dmax);
				Dmin=Dmin-0.2*Dmin;
				Dmax=Dmax+0.2*Dmax;
				stepCurve = (Dmax-Dmin)/double(nStep);
				offset = -Dmin/double(stepCurve);
			}
			calculateCurve_1(Iavg(), projV(), vectorAvg, nStep, stepCurve, offset, Dmin, Dmax);

			/*//DEBUG
			projV.write(formatString("Testprojection_%i_%i.tif", frId, partId));
			//Ipart.write(formatString("particle_%i_%i.tif", frId, partId));
			Iavg.write(formatString("Testaverage_%i_%i.tif", frId, partId));
			//END DEBUG/*/

			if(iterPart->hasNext())
				iterPart->moveNext();

		}//end particles loop


		if(dataInMovie>0){
		calculateCurve_2(projV(), vectorAvg, nStep, slope, intercept, Dmin, Dmax);
			//Vectors to store some results
			it = find(mvIds.begin(), mvIds.end(), (int)m+1);
			if (it == mvIds.end()){
				slopes.push_back(slope);
				intercepts.push_back(intercept);
				mvIds.push_back((int)m+1);
			}
			std::cout << "Estimated curve for movie " << m+1 << ". Slope: " << slope << ". Intercept: " << intercept << std::endl;
		}


		/*if(dataInMovie>0){
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


			std::cout << dataInMovie << " Slopes and intercepts " << dataArray << std::endl;
			//AJ to obtain a soft version of the obtained slopes (dataRow=2)
			calculateBSplineCoeffs(dataArray, boxsize, cij_slopes, xmov, ymov, 2);
			std::cout << dataInMovie << " Slopes coeffs " << cij_slopes << std::endl;
			evaluateBSpline(dataArray, cij_slopes, softArray, xmov, ymov, 2);

			//AJ to obtain a soft version of the obtained interception points (dataRow=3)
			calculateBSplineCoeffs(dataArray, boxsize, cij_intercepts, xmov, ymov, 3);
			std::cout << dataInMovie << " Intercepts coeffs " << cij_intercepts << std::endl;
			evaluateBSpline(dataArray, cij_intercepts, softArray, xmov, ymov, 3);

			std::cout << dataInMovie << " SOFT Slopes and intercepts " << softArray << std::endl;

		}*/

	}//end movie Ids loop



	//NEW part to calculate by ML the shifts of every frame
	/*iterPart->init(mdPart);
	double slope, intercept;
	int position;
	for(int i=0; i<mdPartSize; i++){
		size_t mvId, partId;
		mdPart.getRow(currentRow, iterPart->objId);
		currentRow.getValue(MDL_PARTICLE_ID,partId);
		it = find(particleId.begin(), particleId.end(), (int)partId);
		position=it-particleId.begin();
		slope=slopes[position];
		intercept = intercepts[position];


		if(iterPart->hasNext())
			iterPart->moveNext();
	}*/
	//AJ para generar imagen phantom
	//Image<double> Isim(XSIZE(Ipart()), YSIZE(Ipart()));
	/*if (partId==5818){
		for (int x=0; x<XSIZE(Iavg()); x++){
			for (int y=0; y<YSIZE(Iavg()); y++){
				if(x<(XSIZE(Iavg())/2)-10 || x>(XSIZE(Iavg())/2)+10 || y<(YSIZE(Iavg())/2)-30 || y>(YSIZE(Iavg())/2)+30){
					DIRECT_A2D_ELEM(Isim(), x, y) = 1.2;
					DIRECT_A2D_ELEM(Iproj(), x, y) = 5;
				}else{
					DIRECT_A2D_ELEM(Isim(), x, y) = 1.05;
					DIRECT_A2D_ELEM(Iproj(), x, y) = 20;
				}
			}
		}
		Isim.write("avgNueva.mrc");
		Iproj.write("projNueva.mrc");
		for (int ii=0; ii<24; ii++){
			std::random_device rd;
			std::mt19937 gen(rd());
			Image<int> Isimul(XSIZE(Isim()), YSIZE(Isim()));
			for (int x=0; x<XSIZE(Isim()); x++){
				for (int y=0; y<YSIZE(Isim()); y++){
					double mymean = DIRECT_A2D_ELEM(Isim(), x, y);
					std::poisson_distribution<int> distribution(mymean);
					int number = distribution(gen);
					DIRECT_A2D_ELEM(Isimul(), x, y) = number;
				}
			}
			Isimul.write(formatString("simuladaNueva_%i.mrc",ii));
		}
	}*/
	//FIN AJ phantom//
	/*/AJ para leer phantom y promediar
	Image<double> Isimul;
	Image<double> Itot;
	for (int ii=0; ii<24; ii++){
		Isimul.read(formatString("simuladaNuevaShift_%i.mrc",ii));
		if (ii==0){
			Itot().resize(XSIZE(Isimul()),YSIZE(Isimul()));
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Isimul()){
				DIRECT_MULTIDIM_ELEM(Itot(), n) = DIRECT_MULTIDIM_ELEM(Isimul(), n);
			}
		}
		else{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Isimul()){
				DIRECT_MULTIDIM_ELEM(Itot(), n) += DIRECT_MULTIDIM_ELEM(Isimul(), n);
			}
		}
	}
	Itot()/=24.0;
	Itot.write("totalNueva.mrc");
	//FIN AJ phantom lectura/*/

	/*/AJ to check
	double slope=0, intercept=0;
	Image<double> Inew;
	Inew.read("totalNueva.mrc");
	Iproj.read("projNueva.mrc");
	Inew().setXmippOrigin();
	Iproj().setXmippOrigin();
	calculateCurve(Inew(), Iproj(), slope, intercept);
	std::cerr << ". NUEVA Particle: " << ". Slope: " << slope << ". Intercept: " << intercept <<  std::endl;
	//FIN AJ check/*/

	//ML estimation of shifts
	//-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
	//-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6
	int shiftX[]={-3, -2, -1, 0, 1, 2, 3};
	int shiftY[]={-3, -2, -1, 0, 1, 2, 3};
	MultidimArray<double> lkresults;
	double maxShift = XSIZE(Iavg())/2-10;
	double maxShift2 = maxShift*maxShift;
	//AJ to make a random selection of pixel contributing to the ML estimation
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0,1.0);
	int nRepetitions=1;
	std::vector<int> partIdDone2;
	//double total_lkresults[24];
	//int bestPosX[24];
	//int bestPosY[24];
	double finalPosX[nFrames];
	double finalPosY[nFrames];

	MetaData SFq;
	FileName fnRoot=fnMdMov.insertBeforeExtension("_out_particles");
	FileName fnStackOut=formatString("%s.stk",fnRoot.c_str());
	if(fnStackOut.exists())
		fnStackOut.deleteFile();

	int countOutMd=0;
	for(int m=0; m<nMics; m++){

		iterPart->init(mdPart);
		iterPart2->init(mdPart);
		int countForPart=0;
		int countDisabled=0;
		size_t frId, mvId, partId;

		for(int ii=0; ii<mdPartSize; ii++){

			//Project the volume with the parameters in the image
			double rot, tilt, psi, xValue, yValue;
			bool flip;
			int xcoor, ycoor;
			int enabled;

			mdPart.getRow(currentRow, iterPart->objId);
			currentRow.getValue(MDL_IMAGE,fnPart);
			Ipart.read(fnPart);
			Ipart().setXmippOrigin();
			currentRow.getValue(MDL_ANGLE_ROT,rot);
			currentRow.getValue(MDL_ANGLE_TILT,tilt);
			currentRow.getValue(MDL_ANGLE_PSI,psi);
			currentRow.getValue(MDL_SHIFT_X,xValue);
			currentRow.getValue(MDL_SHIFT_Y,yValue);
			currentRow.getValue(MDL_FLIP,flip);
			currentRow.getValue(MDL_FRAME_ID,frId);
			currentRow.getValue(MDL_MICROGRAPH_ID,mvId);
			currentRow.getValue(MDL_PARTICLE_ID,partId);
			currentRow.getValue(MDL_XCOOR,xcoor);
			currentRow.getValue(MDL_YCOOR,ycoor);
			currentRow.getValue(MDL_ENABLED,enabled);

			for (int hh=0; hh<mvIds.size(); hh++){
				if(mvIds[hh]==mvId){
					slope=slopes[hh];
					intercept=intercepts[hh];
					break;
				}
			}

			if(enabled==-1){
				partIdDone2.push_back((int)partId);
				countDisabled=1;
				if(iterPart->hasNext())
					iterPart->moveNext();
				if(iterPart2->hasNext())
					iterPart2->moveNext();
				continue;
			}

			if(mvId!=m){
				if(iterPart->hasNext())
					iterPart->moveNext();
				if(iterPart2->hasNext())
					iterPart2->moveNext();
				continue;
			}

			it = find(partIdDone2.begin(), partIdDone2.end(), (int)partId);
			if (it != partIdDone2.end()){
				if (countForPart==nFrames-1 || countDisabled==1){
					if(iterPart->hasNext())
						iterPart->moveNext();
					continue;
				}else{
					countForPart++;
				}
			}else{
				countForPart=0;
				countDisabled=0;
			}
			partIdDone2.push_back((int)partId);
			dataInMovie++;

			std::cout << fnPart << std::endl;

			A.initIdentity(3);
			MAT_ELEM(A,0,2)=xValue;
			MAT_ELEM(A,1,2)=yValue;
			if (flip)
			{
				MAT_ELEM(A,0,0)*=-1;
				MAT_ELEM(A,0,1)*=-1;
				MAT_ELEM(A,0,2)*=-1;
			}

			if ((xdim != XSIZE(Ipart())) || (ydim != YSIZE(Ipart())))
				std::cout << "Error: The input particles and volume have different sizes" << std::endl;


			std::cerr << "FOR Movie ID: " << mvId << ". Working with Slope: " << slope << ". Intercept: " << intercept <<  std::endl;
			int estX[nRepetitions];
			int estY[nRepetitions];
			int count;
			for (int zz=0; zz<nRepetitions; zz++){

				lkresults.initZeros(7, 7);
				//Isimul.read(formatString("simuladaNuevaShift_%i.mrc",ii));
				//Isimul().setXmippOrigin();
				for(int jj=0; jj<7; jj++){
					for(int hh=0; hh<7; hh++){
						//Iproj.read("projNueva.mrc");
						//Iproj().setXmippOrigin();
						projectVolume(*projectorV, PV, xdim, xdim,  rot, tilt, psi);
						applyGeometry(LINEAR,projV(),PV(),A,IS_INV,DONT_WRAP,0.);
						projV().setXmippOrigin();

						if (shiftX[jj]!=0 || shiftY[hh]!=0)
							selfTranslate(LINEAR, projV(), vectorR2(shiftX[jj], shiftY[hh]), DONT_WRAP, 0.0);
						double likelihood=0.;
						double lambda, fact;
						count=0;
						for(int n=0; n<YSIZE(Ipart()); n++){
							for(int m=0; m<XSIZE(Ipart()); m++){
								if ((n-round(YSIZE(Ipart())/2))*(n-round(YSIZE(Ipart())/2))+(m-round(XSIZE(Ipart())/2))*(m-round(XSIZE(Ipart())/2))>maxShift2) // continue if the Euclidean distance is too far
									continue;
								//printf("EVAL shift %d %d \n", n, m);
								//double number = distribution(generator);
								//if (number<0.3)
								//	continue;
								count++;
								fact=1.;
								lambda = slope*DIRECT_A2D_ELEM(projV(), n, m)+intercept;
								/*if(DIRECT_A2D_ELEM(projV(), n, m) == 20)
									lambda=1.05;
								else if(DIRECT_A2D_ELEM(projV(), n, m) == 5)
									lambda=1.2;*/
								if (DIRECT_A2D_ELEM(Ipart(), n, m)>0){
									for(int aa=1; aa<=DIRECT_A2D_ELEM(Ipart(), n, m); aa++)
										fact = fact*aa;
								}
								likelihood += -1.*lambda + DIRECT_A2D_ELEM(Ipart(), n, m)*log(lambda) - log(fact);
							}
						}
						DIRECT_A2D_ELEM(lkresults, hh, jj) = likelihood;
						//printf(". Particle. %d. Shift %d, %d. Result: %10.10lf %d \n", ii, shiftX[jj], shiftY[hh], likelihood, count);
						//printf("%d %d %10.10lf \n", shiftX[jj], shiftY[hh], likelihood);
					}
				}
				//std::vector<double>::iterator maxVal;
				//maxVal = std::max_element(lkresults.begin(), lkresults.end());
				//int pos = std::distance(lkresults.begin(), maxVal);
				double sumX[7], sumY[7];
				//Sum by columns (X)
				double maxSumX, maxSumY;
				int posSumX, posSumY;
				for(int jj=0; jj<7; jj++){
					sumX[jj]=0.;
					for(int hh=0; hh<7; hh++){
						sumX[jj]+=DIRECT_A2D_ELEM(lkresults, hh, jj);
					}
					if(jj==0){
						maxSumX= sumX[jj];
						posSumX=0;
					}else{
						if (sumX[jj]>maxSumX){
							maxSumX = sumX[jj];
							posSumX=jj;
						}
					}
				}
				//Sum by rows (Y)
				for(int hh=0; hh<7; hh++){
					sumY[hh]=0.;
					for(int jj=0; jj<7; jj++){
						sumY[hh]+=DIRECT_A2D_ELEM(lkresults, hh, jj);
					}
					if(hh==0){
						maxSumY= sumY[hh];
						posSumY=0;
					}else{
						if (sumY[hh]>maxSumY){
							maxSumY = sumY[hh];
							posSumY=hh;
						}
					}
				}

				MultidimArray<double> lkresults_copy(lkresults);
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(lkresults)
					DIRECT_MULTIDIM_ELEM(lkresults_copy,n) = DIRECT_MULTIDIM_ELEM(lkresults,n);

				//Image<double> lk;
				//lk()=lkresults;
				//lk.write(formatString("lkresults_%i.mrc",ii));
				double estXAux, estYAux;
				bestShift(lkresults, estXAux, estYAux, NULL, -1);
				int bestPosX = (int)round(estXAux);
				if(bestPosX>6)
					bestPosX=6;
				else if(bestPosX<0)
					bestPosX=0;
				int bestPosY = (int)round(estYAux);
				if(bestPosY>6)
					bestPosY=6;
				else if(bestPosY<0)
					bestPosY=0;
				printf(". BEST POS for particle %d. Shift %lf, %lf, %d, %d \n", countForPart, estXAux, estYAux, shiftX[bestPosX], shiftY[bestPosY]);
				//printf("New estimation marginalized %d %d \n", shiftX[posSumX], shiftY[posSumY]);
				//total_lkresults[ii]=DIRECT_A2D_ELEM(lkresults_copy, (int)estYAux, (int)estXAux);
				estX[zz]=shiftX[bestPosX];
				estY[zz]=shiftY[bestPosY];

				//AJ to test with correlation
				projectVolume(*projectorV, PV, xdim, xdim,  rot, tilt, psi);
				applyGeometry(LINEAR,projV(),PV(),A,IS_INV,DONT_WRAP,0.);
				projV().setXmippOrigin();
				MultidimArray< std::complex<double> > fftIproj, fftIsimul;
				MultidimArray<double> Mcorr;
				Mcorr.resize(YSIZE(Ipart()), XSIZE(Ipart()));
				Mcorr.setXmippOrigin();
				CorrelationAux aux;
				// Generate mask
				//MultidimArray<int> mask;
				//mask.resize(YSIZE(Isimul()), XSIZE(Isimul()));
				//mask.initConstant(1);
				//mask.setXmippOrigin();
				Mask mask;
				mask.type = BINARY_CIRCULAR_MASK;
				mask.mode = INNER_MASK;
				size_t rad = (size_t)std::min(XSIZE(Ipart())*0.5, YSIZE(Ipart())*0.5);
				mask.R1 = rad;
				mask.resize(YSIZE(Ipart()), XSIZE(Ipart()));
				mask.get_binary_mask().setXmippOrigin();
				mask.generate_mask();
				FourierTransformer transformer;

				//projV.write(formatString("VeamosProjV.tif"));
				//Ipart.write(formatString("VeamosIpart.tif"));

				/*transformer.FourierTransform(projV(), fftIproj);
				transformer.FourierTransform(Ipart(), fftIsimul);
				correlation_matrix(fftIproj, fftIsimul, Mcorr, aux);
				double sX, sY;
				double avg, stddev, min_val, max_val;
				computeStats_within_binary_mask(mask.get_binary_mask(), Mcorr, min_val, max_val, avg, stddev);
				bestShift(Mcorr, sX, sY, &mask.get_binary_mask(), 6);
				//Image<double> correlation;
				//correlation()=Mcorr;
				//correlation.write(formatString("correlation_%i.mrc",ii));
				printf(". BEST CORR for particle %d. STATS: %lf, %lf, %lf, %lf. SHIFT %lf, %lf \n", ii, min_val, max_val, avg, stddev, sX, sY);*/

			}//end loop zz

		//printf("BEST POS for particle %d  \n", ii);
		double sumXX=0., sumYY=0.;
		for (int zz=0; zz<nRepetitions; zz++){
			//printf("	%d, %d \n", estX[zz], estY[zz]);
			sumXX+=estX[zz];
			sumYY+=estY[zz];
		}
		finalPosX[countForPart]=sumXX/nRepetitions;
		finalPosY[countForPart]=sumYY/nRepetitions;
		//printf("MEAN values: %d, %lf, %lf\n", countForPart, finalPosX[countForPart], finalPosY[countForPart]);

		if (countForPart==nFrames-1){


			PseudoInverseHelper pseudoInverter;
			Matrix2D<double> &X = pseudoInverter.A;
			Matrix1D<double> &y = pseudoInverter.b;
			Matrix1D<double> resultX(nFrames);
			Matrix1D<double> resultY(nFrames);
			X.resizeNoCopy(nFrames,3);
			y.resizeNoCopy(nFrames);
			for(int jj=0; jj<24; jj++){
				X(jj,0)=1;
				X(jj,1)=jj+1;
				X(jj,2)=(jj+1)*(jj+1);
				//X(ii,3)=(ii+1)*(ii+1)*(ii+1);
				y(jj)=finalPosX[jj];
			}
			Matrix1D<double> alpha(3);
			solveLinearSystem(pseudoInverter, alpha);
			printf("SOLVING LINEAR SYSTEM FOR X \n");
			printf("alpha(0)=%lf \n", alpha(0));
			printf("alpha(1)=%lf \n", alpha(1));
			printf("alpha(2)=%lf \n", alpha(2));
			//printf("alpha(3)=%lf \n", alpha(3));
			matrixOperation_Ax(X, alpha, resultX);

			y.resizeNoCopy(nFrames);
			for(int jj=0; jj<nFrames; jj++){
				y(jj)=finalPosY[jj];
			}
			solveLinearSystem(pseudoInverter, alpha);
			printf("SOLVING LINEAR SYSTEM FOR Y \n");
			printf("alpha(0)=%lf \n", alpha(0));
			printf("alpha(1)=%lf \n", alpha(1));
			printf("alpha(2)=%lf \n", alpha(2));
			//printf("alpha(3)=%lf \n", alpha(3));
			matrixOperation_Ax(X, alpha, resultY);

			FileName fnTest;
			Ifinal().resize(Ipart());
			Ifinal().initZeros();
			for(int mm=0; mm<nFrames; mm++){
				mdPart.getRow(currentRow2, iterPart2->objId);
				currentRow2.getValue(MDL_IMAGE,fnTest);
				IpartOut.read(fnTest);
				IpartOut().setXmippOrigin();
				Aout.initIdentity(3);
				MAT_ELEM(Aout,0,2)=resultX(mm);
				MAT_ELEM(Aout,1,2)=resultY(mm);
				applyGeometry(LINEAR,Iout(),IpartOut(),A,IS_NOT_INV,DONT_WRAP,0.);
				Iout().setXmippOrigin();
				Ifinal()+=Iout();
				//double valueX=xValue+resultX(mm);
				//double valueY=yValue+resultY(mm);
				//currentRow2.setValue(MDL_SHIFT_X, valueX);
				//currentRow2.setValue(MDL_SHIFT_Y, valueY);
				//SFq.addRow(currentRow2);
				if(iterPart2->hasNext())
					iterPart2->moveNext();
			}
			//To invert the contrast
			/*double DminFinal, DmaxFinal, v, temp;
			Ifinal().computeDoubleMinMax(DminFinal, DmaxFinal);
			double irange=1.0/(Dmax - Dmin);
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Ifinal()){
				if (v < 1)
					temp = v;
				else
				    temp = log10(v);
				v=DIRECT_MULTIDIM_ELEM(Ifinal(),n);
				DIRECT_MULTIDIM_ELEM(Ifinal(),n) = (Dmax - v) * irange;
			}*/
			Ifinal.write(fnStackOut,countOutMd,true,WRITE_APPEND);
			countOutMd++;
			FileName fnToSave;
			fnToSave.compose(countOutMd, fnStackOut);
			size_t id = SFq.addObject();
			SFq.setValue(MDL_IMAGE, fnToSave, id);

		}

		if(iterPart->hasNext())
			iterPart->moveNext();

		}//end loop particles

	}//end loop movies

	FileName fnOut;
	fnOut = fnMdMov.insertBeforeExtension("_out");
	printf("Writing output metadata \n");
	printf("%s \n ", fnOut.getString().c_str());
	SFq.write(fnOut);


	exit(0);


	//FIN AJ ML


	/////////////////////////
	//Second Part
	/////////////////////////
	MultidimArray<double> matrixWeights, maxvalues;
	double cutfreq;
	double inifreq=0.5; //0.05;
	double step=0.05; //0.05;
	int Nsteps= int((0.5-inifreq)/step);
	double maxvalue=-1.;
	size_t mvId, frId, partId;
	Iavg().resize(XSIZE(Ipart()), YSIZE(Ipart()));


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
			/*if (prevMvId!=-1){
				std::cerr << " maxvalue " << maxvalue << std::endl;
				std::cerr << "  matrixWeights " << matrixWeights << std::endl;
				matrixWeights/=maxvalue;
				std::cerr << " NORM matrixWeights " << matrixWeights << std::endl;
				countMv++;

				for (int s=0; s<ZSIZE(matrixWeights); s++){
					Matrix2D<double> matrixWeightsMat;
					MultidimArray<double> matrixWeightsSlice;
					matrixWeights.getSlice(s, matrixWeightsSlice);
					std::cerr << "- Slice: " << XSIZE(matrixWeightsSlice) << " " << YSIZE(matrixWeightsSlice) << " " << ZSIZE(matrixWeightsSlice) << std::endl;
					matrixWeightsSlice.copy(matrixWeightsMat);
					std::cerr << "- Matrix: " << matrixWeightsMat.Xdim() << " " << matrixWeightsMat.Ydim() << " " << matrixWeightsMat << std::endl;
					Matrix2D<double> U,V;
					Matrix1D<double> S;
					svdcmp(matrixWeightsMat,U,S,V);
					std::cerr << "- SVD: " << U << std::endl;
					//AJ prueba
					Matrix2D<double> Smat;
					Smat.initZeros(S.size(),S.size());
					for (int h=0; h<S.size(); h++){
						if (h<2)
						    Smat(h,h)=S(h);
					}
					Matrix2D<double> result1, result;
					matrixOperation_AB(U, Smat, result1);
					matrixOperation_ABt(result1, V, result);
					std::cerr << "- SVD recons: " << result << std::endl;
				}
			}*/
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

		for(int n=0; n<Nsteps+1; n++){

			std::cerr << "- Particle: " << partId << " and frequency " << n << std::endl;

				/*Ipart.read(fnPart);
				Ipart().setXmippOrigin();

				projectVolume(*projectorV, PV, xdim, xdim,  rot, tilt, psi);
				applyGeometry(LINEAR,projV(),PV(),A,IS_INV,DONT_WRAP,0.);
				projV().setXmippOrigin();
				//TODO: ver si estamos alineando en sentido correcto la proyeccion - hecho en el debugging en el produceSideInfo

				//To invert contrast in the projections
				double Dmin, Dmax, irange, val;
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(projV()){
					val=DIRECT_MULTIDIM_ELEM(projV(),n);
					DIRECT_MULTIDIM_ELEM(projV(),n) = (Dmax - val) * irange;
				}

				//filtering the projections with the ctf
				ctf.readFromMdRow(currentRow);
				ctf.produceSideInfo();
				ctf.applyCTF(projV(), samplingRate, false);

				//averaging movie particle image with the ones in all the frames but without the current one
				averagingAll(mdPart, Ipart(), Iavg(), partId, frId, mvId, true);
				//TODO check how we have to average the movie particles to calculate the correlations
				*/

				//AJ para leer phantom y promediar
				Image<double> Itot, Imean;
				Iproj.read("projNueva.mrc");
				Iproj().setXmippOrigin();
				Itot().initZeros(XSIZE(Iproj()),YSIZE(Iproj()));
				Imean().initZeros(XSIZE(Iproj()),YSIZE(Iproj()));
				if (partId==5818){
					Image<int> Isimul;
					int count=0;
					for (int ii=0; ii<24; ii++){
						Isimul.read(formatString("simuladaNuevaShift_%i.mrc", ii));
						if (frId==ii+1){
							FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Isimul()){
								DIRECT_MULTIDIM_ELEM(Imean(), n) += (double)DIRECT_MULTIDIM_ELEM(Isimul(), n);
							}
							continue;
						}
						FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Isimul()){
							DIRECT_MULTIDIM_ELEM(Itot(), n) += (double)DIRECT_MULTIDIM_ELEM(Isimul(), n);
							DIRECT_MULTIDIM_ELEM(Imean(), n) += (double)DIRECT_MULTIDIM_ELEM(Isimul(), n);
						}
						count++;
					}
					Itot()/=23.0;
					Itot.write(formatString("total_%i.mrc", 24));
					Imean()/=24.0;
					Imean.write(formatString("mean_%i.mrc", 24));
					double meanD, stdD;
					Imean().computeAvgStdev(meanD,stdD);
					Iavg().initZeros(XSIZE(Isimul()), YSIZE(Isimul()));
					projV().initZeros(XSIZE(Isimul()), YSIZE(Isimul()));
					FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Itot()){
						DIRECT_MULTIDIM_ELEM(Iavg(), n) = (double)DIRECT_MULTIDIM_ELEM(Itot(), n);
					}
					FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Iproj()){
						DIRECT_MULTIDIM_ELEM(projV(), n) = (double)DIRECT_MULTIDIM_ELEM(Iproj(), n);
					}
					//Iavg().setXmippOrigin();
					//projV().setXmippOrigin();
					double Dmin, Dmax, irange, val;
					projV().computeDoubleMinMax(Dmin, Dmax);
					irange=1.0/(Dmax - Dmin);
					FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(projV()){
						val=DIRECT_MULTIDIM_ELEM(projV(),n);
						DIRECT_MULTIDIM_ELEM(projV(),n) = (Dmax - val) * irange;
					}
					Iavg.write(formatString("myAvg.mrc"));
					projV.write(formatString("myProj.mrc"));

				//FIN AJ phantom lectura//

			cutfreq = inifreq + step*n;


			//filtering the projected particles with the lowpass filter
			Filter.w1=cutfreq;
			Filter.generateMask(projV());
			Filter.applyMaskSpace(projV());

			//filtering the averaged movie particle (leaving out the j-frame)
			Filter.generateMask(Iavg());
			Filter.applyMaskSpace(Iavg());

			//filtering the averaged movie particle (all the frames)
			Filter.generateMask(Imean());
			Filter.applyMaskSpace(Imean());

			Iavg.write(formatString("myAvgFiltered.mrc"));
			projV.write(formatString("myProjFiltered.mrc"));


			//calculate similarity measures between averaged movie particles and filtered projection
			double weight;
			double corrNRef, corrMRef, corrWRef, imedRef;
			similarity(projV(), Imean(), corrNRef, corrMRef, corrWRef, imedRef);
			printf("REF CORRELATION %8.20lf \n", corrNRef);
			double corrN, corrM, corrW, imed;
			similarity(projV(), Iavg(), corrN, corrM, corrW, imed, meanD);
			//TODO: la corrW que viene de Idiff no tiene mucho sentido porque la Idiff en este caso no nos dice nada

			/*/DEBUG
			if(frId==nFrames){
			projV.write(formatString("projection_%i_%i.tif", frId, partId));
			Ipart.write(formatString("particle_%i_%i.tif", frId, partId));
			Iavg.write(formatString("average_%i_%i.tif", frId, partId));
			}
			//END DEBUG/*/

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

				}

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


	std::cerr << " maxvalue " << maxvalue << std::endl;
	//std::cerr << "  matrixWeights " << matrixWeights << std::endl;
	printf("matrixWeights \n");
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(matrixWeights){
		printf("%8.20lf \n", DIRECT_MULTIDIM_ELEM(matrixWeights,n));
	}
	matrixWeights/=maxvalue;
	std::cerr << " NORM matrixWeights " << matrixWeights << std::endl;
	countMv++;

	for (int s=0; s<ZSIZE(matrixWeights); s++){
		Matrix2D<double> matrixWeightsMat;
		MultidimArray<double> matrixWeightsSlice;
		matrixWeights.getSlice(s, matrixWeightsSlice);
		std::cerr << "- Slice: " << XSIZE(matrixWeightsSlice) << " " << YSIZE(matrixWeightsSlice) << " " << ZSIZE(matrixWeightsSlice) << std::endl;
		matrixWeightsSlice.copy(matrixWeightsMat);
		std::cerr << "- Matrix: " << matrixWeightsMat.Xdim() << " " << matrixWeightsMat.Ydim() << " " << matrixWeightsMat << std::endl;
		Matrix2D<double> U,V;
		Matrix1D<double> S;
		svdcmp(matrixWeightsMat,U,S,V);
		std::cerr << "- SVD: " << U << std::endl;
		//AJ prueba
		Matrix2D<double> Smat;
		Smat.initZeros(S.size(),S.size());
		for (int h=0; h<S.size(); h++){
			if (h<2)
				Smat(h,h)=S(h);
		}
		Matrix2D<double> result1, result;
		matrixOperation_AB(U, Smat, result1);
		matrixOperation_ABt(result1, V, result);
		std::cerr << "- SVD recons: " << result << std::endl;
	}



	//MultidimArray<double> weightsperfreq;
	//weightsperfreq.initZeros(NSIZE(matrixWeights), ZSIZE(matrixWeights), YSIZE(matrixWeights), XSIZE(matrixWeights));
	//calculateFrameWeightPerFreq(matrixWeights, weightsperfreq, maxvalues);

}



