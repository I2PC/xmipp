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

	//MOVIE PARTICLES IMAGES
	MetaData mdPartPrev, mdPart;
	MDRow currentRow;
	mdPartPrev.read(fnMdMov,NULL);
	mdPartSize = mdPartPrev.size();
	mdPart.sort(mdPartPrev, MDL_PARTICLE_ID);

	MDIterator *iterPart = new MDIterator();

	double rot, tilt, psi, x, y;
	bool flip;
	size_t frId, mvId, partId;
	int xcoor, ycoor;
	int enabled;
	FileName fnPart;
	String aux;
	ctfs = new CTFDescription[mdPartSize];

	iterPart->init(mdPart);

	for(int i=0; i<mdPartSize; i++){

		//Project the volume with the parameters in the image
		mdPart.getRow(currentRow, iterPart->objId);
		currentRow.getValue(MDL_IMAGE,fnPart);
		currentRow.getValue(MDL_PARTICLE_ID,partId);
		currentRow.getValue(MDL_MICROGRAPH_ID,mvId);
		currentRow.getValue(MDL_FRAME_ID,frId);
		currentRow.getValue(MDL_ANGLE_ROT,rot);
		currentRow.getValue(MDL_ANGLE_TILT,tilt);
		currentRow.getValue(MDL_ANGLE_PSI,psi);
		currentRow.getValue(MDL_SHIFT_X,x);
		currentRow.getValue(MDL_SHIFT_Y,y);
		currentRow.getValue(MDL_FLIP,flip);
		currentRow.getValue(MDL_XCOOR,xcoor);
		currentRow.getValue(MDL_YCOOR,ycoor);
		currentRow.getValue(MDL_ENABLED,enabled);
		ctfs[i].readFromMdRow(currentRow);

		objIds.push_back((int)iterPart->objId);
		fnParts.push_back((std::string)fnPart.getString());
		partIds.push_back((int)partId);
		mvIds.push_back((int)mvId);
		frIds.push_back((int)frId);
		rots.push_back(rot);
		tilts.push_back(tilt);
		psis.push_back(psi);
		xs.push_back(x);
		ys.push_back(y);
		flips.push_back(flip);
		xcoors.push_back(xcoor);
		ycoors.push_back(ycoor);
		enables.push_back(enabled);


		if(iterPart->hasNext())
			iterPart->moveNext();

	}


}

void ProgParticlePolishing::averagingAll(const MetaData &mdPart, const MultidimArray<double> &I, MultidimArray<double> &Iout, size_t partId, size_t frameId, size_t movieId, bool noCurrent){

	//MDRow currentRow;
	FileName fnPart;
	Image<double> Ipart;
	size_t newPartId, newFrameId, newMovieId;
	//size_t mdPartSize = mdPart.size();
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
	//MDIterator *iterPart = new MDIterator(mdPart);
	int frameIdI = int(frameId);
	for(int i=0; i<mdPartSize; i++){

		//mdPart.getRow(currentRow, iterPart->objId);
		//currentRow.getValue(MDL_PARTICLE_ID, newPartId);
		//currentRow.getValue(MDL_MICROGRAPH_ID, newMovieId);
		newPartId = partIds[i];
		newMovieId = mvIds[i];
		if((newPartId==partId) && (newMovieId==movieId)){
			//currentRow.getValue(MDL_FRAME_ID, newFrameId);
			newFrameId = frIds[i];
			int newFrameIdI = int(newFrameId);
			if(newFrameIdI==frameIdI){
				//if(iterPart->hasNext())
				//	iterPart->moveNext();
				continue;
			}
			else{ //averaging with all the frames
				//currentRow.getValue(MDL_IMAGE,fnPart);
				fnPart = (String)fnParts[i];
				Ipart.read(fnPart);
				Ipart().setXmippOrigin();
				Iout+=Ipart();
				count+=1.0;
			}
		}
		else{
			//if(iterPart->hasNext())
			//	iterPart->moveNext();
			continue;
		}

		//if(iterPart->hasNext())
		//	iterPart->moveNext();

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


void ProgParticlePolishing::writingOutput(){

	//MOVIE PARTICLES IMAGES
	MetaData mdPartPrev, mdPart;
	MDRow currentRow;
	mdPartPrev.read(fnMdMov,NULL);
	mdPartSize = mdPartPrev.size();
	mdPart.sort(mdPartPrev, MDL_PARTICLE_ID);

	MDIterator *iterPart = new MDIterator();

	double rot, tilt, psi, x, y;
	bool flip;
	size_t frId, mvId, partId;
	int xcoor, ycoor;
	int enabled;
	FileName fnPart;
	String aux;

	MetaData SFq2;
	FileName fnRoot2=fnMdMov.insertBeforeExtension("_out_particles");
	FileName fnStackOut2=formatString("%s.stk",fnRoot2.c_str());
	if(fnStackOut2.exists())
		fnStackOut2.deleteFile();

	iterPart->init(mdPart);

	int partIdPrev=-1;
	Image<double> Ipart, Ifinal;
	int countForPart=0;
	int countOutMd=0;
	for(int i=0; i<mdPartSize; i++){

		//Project the volume with the parameters in the image
		mdPart.getRow(currentRow, iterPart->objId);
		currentRow.getValue(MDL_PARTICLE_ID,partId);
		currentRow.getValue(MDL_ENABLED,enabled);

		if(enabled==-1){
			continue;
		}

		if(partIdPrev!=partId){
			countForPart=0;
			Ifinal().resize(Ipart());
			Ifinal().initZeros();
			partIdPrev=partId;
		}
		currentRow.getValue(MDL_IMAGE,fnPart);
		Ipart.read(fnPart);
		Ipart().setXmippOrigin();
		selfTranslate(NEAREST, Ipart(), vectorR2(round(resultShiftX[i]), round(resultShiftY[i])), DONT_WRAP, 0.0);
		Ifinal()+=Ipart();
		countForPart++;

		if (countForPart==nFrames-1){
			Ifinal.write(fnStackOut2,countOutMd,true,WRITE_APPEND);
			countOutMd++;
			FileName fnToSave2;
			fnToSave2.compose(countOutMd, fnStackOut2);
			//size_t id = SFq.addObject();
			//SFq.setValue(MDL_IMAGE, fnToSave, id);
			currentRow.setValue(MDL_IMAGE, fnToSave2);
			SFq2.addRow(currentRow);
		}

	} //end mdPartSize loop

	FileName fnOut2;
	fnOut2 = fnMdMov.insertBeforeExtension("_out");
	printf("Writing output metadata\n");
	printf("%s \n ", fnOut2.getString().c_str());
	SFq2.write(fnOut2);

}



void ProgParticlePolishing::run()
{

	produceSideInfo();

	//MOVIE PARTICLES IMAGES
	MetaData mdPartPrev, mdPart;
	//size_t Xdim, Ydim, Zdim, Ndim;
	//mdPartPrev.read(fnMdMov,NULL);
	//size_t mdPartSize = mdPartPrev.size();

	//mdPart.sort(mdPartPrev, MDL_PARTICLE_ID);

	MDIterator *iterPart = new MDIterator();
	MDIterator *iterPart2 = new MDIterator();
	FileName fnPart;
	Image<double> Ipart, projV, Iavg, Iproj, IpartOut, Iout, Ifinal, IfinalAux;
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


	/////////////////////////
	//First Part
	/////////////////////////
	std::vector<int> partIdDone;
	std::vector<int>::iterator it;
	int dataInMovie;
	std::vector<int> mvIdsAux;
	std::vector<double> slopes;
	std::vector<double> intercepts;
	double slope=0., intercept=0.;
	int nStep=30;
	MultidimArray<double> vectorAvg;

	double Dmin=0., Dmax=0.;
	double stepCurve;
	double offset;


	for(int m=1; m<=nMics; m++){

		double rot, tilt, psi, x, y;
		bool flip;
		size_t frId, mvId, partId;
		int xcoor, ycoor;
		int enabled;

		//iterPart->init(mdPart);
		dataInMovie = 0;
		vectorAvg.initZeros(2, nStep);
		slope=0.;
		intercept=0.;

		for(int i=0; i<mdPartSize; i++){

			/*
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
			*/

			frId = frIds[i];
			mvId = mvIds[i];
			partId = partIds[i];
			enabled = enables[i];


			//std::cout << mvId << " " << frId << " " << partId << std::endl;
			if(enabled==-1){
				partIdDone.push_back((int)partId);
				//if(iterPart->hasNext())
				//	iterPart->moveNext();
				continue;
			}

			if(mvId!=m){
				//if(iterPart->hasNext())
				//	iterPart->moveNext();
				continue;
			}

			it = find(partIdDone.begin(), partIdDone.end(), (int)partId);
			if (it != partIdDone.end()){
				//if(iterPart->hasNext())
				//	iterPart->moveNext();
				continue;
			}
			partIdDone.push_back((int)partId);
			dataInMovie++;

			fnPart = (String)fnParts[i];
			Ipart.read(fnPart);
			Ipart().setXmippOrigin();
			rot = rots[i];
			tilt=tilts[i];
			psi=psis[i];
			x = xs[i];
			y = ys[i];
			flip = flips[i];
			xcoor = xcoors[i];
			ycoor = ycoors[i];

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

			//if(iterPart->hasNext())
			//	iterPart->moveNext();

		}//end particles loop


		if(dataInMovie>0){
			calculateCurve_2(projV(), vectorAvg, nStep, slope, intercept, Dmin, Dmax);
			//Vectors to store some results
			it = find(mvIdsAux.begin(), mvIdsAux.end(), m);
			if (it == mvIdsAux.end()){
				slopes.push_back(slope);
				intercepts.push_back(intercept);
				mvIdsAux.push_back(m);
			}
			std::cout << "Estimated curve for movie " << m << ". Slope: " << slope << ". Intercept: " << intercept << std::endl;
		}


	}//end movie Ids loop


	//ML estimation of shifts
	//-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
	//-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6
	int shiftX[]={-3, -2, -1, 0, 1, 2, 3};
	int shiftY[]={-3, -2, -1, 0, 1, 2, 3};
    int myL = (int)(sizeof(shiftX)/sizeof(*shiftX));
	MultidimArray<double> lkresults;
	double maxShift = XSIZE(Iavg())/2-10;
	double maxShift2 = maxShift*maxShift;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0,1.0);
	std::vector<int> partIdDone2;
	double finalPosX[nFrames];
	double finalPosY[nFrames];

	MetaData SFq;
	FileName fnRoot=fnMdMov.insertBeforeExtension("_out_particles");
	FileName fnStackOut=formatString("%s.stk",fnRoot.c_str());
	if(fnStackOut.exists())
		fnStackOut.deleteFile();


		//iterPart->init(mdPart);
		//iterPart2->init(mdPart);
		int countForPart=0;
		int countDisabled=0;
		size_t frId, mvId, partId;

		for(int ii=0; ii<mdPartSize; ii++){

			//Project the volume with the parameters in the image
			double rot, tilt, psi, xValue, yValue;
			bool flip;
			int xcoor, ycoor;
			int enabled;

			/*
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
			*/

			frId = frIds[ii];
			mvId = mvIds[ii];
			partId = partIds[ii];
			enabled = enables[ii];


			for (int hh=0; hh<mvIdsAux.size(); hh++){
				if(mvIdsAux[hh]==mvId){
					slope=slopes[hh];
					intercept=intercepts[hh];
					break;
				}
			}

			if(enabled==-1){
				partIdDone2.push_back((int)partId);
				countDisabled=1;
				//if(iterPart->hasNext())
				//	iterPart->moveNext();
				//if(iterPart2->hasNext())
				//	iterPart2->moveNext();
				continue;
			}


			it = find(partIdDone2.begin(), partIdDone2.end(), (int)partId);
			if (it != partIdDone2.end()){
				if (countDisabled==1){
					//if(iterPart->hasNext())
					//	iterPart->moveNext();
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

			fnPart = (String)fnParts[ii];
			Ipart.read(fnPart);
			Ipart().setXmippOrigin();
			rot = rots[ii];
			tilt=tilts[ii];
			psi=psis[ii];
			xValue = xs[ii];
			yValue = ys[ii];
			flip = flips[ii];
			xcoor = xcoors[ii];
			ycoor = ycoors[ii];

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
			int estX;
			int estY;

			lkresults.initZeros(myL, myL);
			for(int jj=0; jj<myL; jj++){
				for(int hh=0; hh<myL; hh++){
					projectVolume(*projectorV, PV, xdim, xdim,  rot, tilt, psi);
					applyGeometry(LINEAR,projV(),PV(),A,IS_INV,DONT_WRAP,0.);
					projV().setXmippOrigin();

					if (shiftX[jj]!=0 || shiftY[hh]!=0)
						selfTranslate(LINEAR, projV(), vectorR2((double)shiftX[jj], (double)shiftY[hh]), DONT_WRAP, 0.0);
					double likelihood=0.;
					double lambda, fact;
					for(int n=0; n<YSIZE(Ipart()); n++){
						for(int m=0; m<XSIZE(Ipart()); m++){
							if ((n-round(YSIZE(Ipart())/2))*(n-round(YSIZE(Ipart())/2))+(m-round(XSIZE(Ipart())/2))*(m-round(XSIZE(Ipart())/2))>maxShift2) // continue if the Euclidean distance is too far
								continue;
							fact=1.;
							lambda = slope*DIRECT_A2D_ELEM(projV(), n, m)+intercept;
							if (DIRECT_A2D_ELEM(Ipart(), n, m)>0){
								for(int aa=1; aa<=DIRECT_A2D_ELEM(Ipart(), n, m); aa++)
									fact = fact*aa;
							}
							likelihood += -1.*lambda + DIRECT_A2D_ELEM(Ipart(), n, m)*log(lambda) - log(fact);
						}
					}
					DIRECT_A2D_ELEM(lkresults, hh, jj) = likelihood;
				}
			}


			MultidimArray<double> lkresults_copy(lkresults);
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(lkresults)
				DIRECT_MULTIDIM_ELEM(lkresults_copy,n) = DIRECT_MULTIDIM_ELEM(lkresults,n);

			double estXAux, estYAux;
			bestShift(lkresults, estXAux, estYAux, NULL, -1);
			int bestPosX = (int)round(estXAux);
			if(bestPosX>myL-1)
				bestPosX=myL-1;
			else if(bestPosX<0)
				bestPosX=0;
			int bestPosY = (int)round(estYAux);
			if(bestPosY>myL-1)
				bestPosY=myL-1;
			else if(bestPosY<0)
				bestPosY=0;
			printf(". BEST POS for particle %d. Shift %d, %d \n", countForPart, shiftX[bestPosX], shiftY[bestPosY]);

			finalPosX[countForPart]=shiftX[bestPosX];
			finalPosY[countForPart]=shiftY[bestPosY];

			if (countForPart==nFrames-1){

				PseudoInverseHelper pseudoInverter;
				Matrix2D<double> &X = pseudoInverter.A;
				Matrix1D<double> &y = pseudoInverter.b;
				Matrix1D<double> resultX(nFrames);
				Matrix1D<double> resultY(nFrames);
				X.resizeNoCopy(nFrames,3);
				y.resizeNoCopy(nFrames);
				for(int jj=0; jj<nFrames; jj++){
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

				//Ifinal().resize(Ipart());
				//Ifinal().initZeros();
				for(int mm=0; mm<nFrames; mm++){
					if(resultShiftX.size() != (ii-(nFrames-1)+mm))
						std::cerr << "Error saving data" << std::endl;
					else{
						resultShiftX.push_back(resultX(mm));
						resultShiftY.push_back(resultY(mm));
						printf("Particle %d frame %d shiftX %lf shiftY %lf \n", ii, mm, resultX(mm), resultY(mm));
					}
					//selfTranslate(NEAREST, Ipart(), vectorR2(round(resultX(mm)), round(resultY(mm))), DONT_WRAP, 0.0);
					//Ifinal()+=Ipart();
				}

			}

			//if(iterPart->hasNext())
				//iterPart->moveNext();

		}//end loop particles


	/*FileName fnOut;
	fnOut = fnMdMov.insertBeforeExtension("_out");
	printf("Writing output metadata \n");
	printf("%s \n ", fnOut.getString().c_str());
	SFq.write(fnOut);*/




	exit(0);


	//FIN AJ ML


	/////////////////////////
	//Second Part
	/////////////////////////

	mdPartPrev.read(fnOut,NULL);
	mdPartSize = mdPartPrev.size();
	mdPart.sort(mdPartPrev, MDL_MICROGRAPH_ID);

	MultidimArray<double> matrixWeights, maxvalues;
	FourierFilter FilterBP, FilterLP;
	FilterBP.FilterBand=BANDPASS;
	FilterBP.FilterShape=RAISED_COSINE;
	FilterLP.FilterBand=LOWPASS;
	FilterLP.FilterShape=RAISED_COSINE;
	double cutfreq;
	//double inifreq=0.25; //0.05;
	//double step=0.25; //0.05;
	//int Nsteps= int((0.5-inifreq)/step);
	int Nsteps=1;
	double bandSize=0.5/(double)Nsteps;
	double maxvalue=-1.;
	//size_t mvId, frId, partId;
	Iavg().resize(XSIZE(Ipart()), YSIZE(Ipart()));


	iterPart->init(mdPart);
	std::vector<int> numPartPerMov;
	int prevMvId=-1, prevPartId=-1, partCount=0;
	for(int i=0; i<mdPartSize; i++){
		size_t mvId, partId;
		int enabled;
		mdPart.getRow(currentRow, iterPart->objId);
		currentRow.getValue(MDL_MICROGRAPH_ID,mvId);
		currentRow.getValue(MDL_PARTICLE_ID,partId);
		currentRow.getValue(MDL_ENABLED,enabled);
		if(enabled==-1){
			if(iterPart->hasNext())
				iterPart->moveNext();
			continue;
		}
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

	int countOutMd=0;
	prevMvId=-1;
	prevPartId=-1;
	partCount=0;
	int countMv=-1;
	iterPart->init(mdPart);
	iterPart2->init(mdPart);
	Image<double> Ipartaux, projVaux;
	Matrix2D<double> resultWeights;

	MetaData SFq2;
	FileName fnRoot2=fnMdMov.insertBeforeExtension("_out_particles_filtered");
	FileName fnStackOut2=formatString("%s.stk",fnRoot2.c_str());
	if(fnStackOut2.exists())
		fnStackOut2.deleteFile();

	double totalWeight[Nsteps];
	for(int i=0; i<mdPartSize; i++){
		
		//Project the volume with the parameters in the image
		double rot, tilt, psi, x, y;
		bool flip;
		size_t frId, mvId, partId;
		int xcoor, ycoor;
		int enabled;

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

		for (int hh=0; hh<mvIds.size(); hh++){
			if(mvIds[hh]==mvId){
				slope=slopes[hh];
				intercept=intercepts[hh];
				break;
			}
		}

		if(enabled==-1){
			if(iterPart->hasNext())
				iterPart->moveNext();
			if(iterPart2->hasNext())
				iterPart2->moveNext();
			continue;
		}

		if(prevMvId!=mvId){
			countMv++;
			partCount=0;
			prevMvId=mvId;
			maxvalue=-1.;
			matrixWeights.initZeros(numPartPerMov[countMv], nFrames, Nsteps);
		}
		if(partId!=prevPartId){
			prevPartId=partId;
			partCount++;
			for(int n=0; n>Nsteps; n++)
				totalWeight[n]=0.;
		}

		A.initIdentity(3);
		MAT_ELEM(A,0,2)=x;
		MAT_ELEM(A,1,2)=y;
		if (flip){
			MAT_ELEM(A,0,0)*=-1;
			MAT_ELEM(A,0,1)*=-1;
			MAT_ELEM(A,0,2)*=-1;
		}

		for(int n=0; n<Nsteps; n++){

			//std::cerr << "- Particle: " << partId << " and frequency " << n << std::endl;

				Ipart.read(fnPart);
				Ipart().setXmippOrigin();

				projectVolume(*projectorV, PV, xdim, xdim,  rot, tilt, psi);
				applyGeometry(LINEAR,projV(),PV(),A,IS_INV,DONT_WRAP,0.);
				projV().setXmippOrigin();
				//TODO: ver si estamos alineando en sentido correcto la proyeccion - hecho en el debugging en el produceSideInfo

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
				averagingAll(mdPart, Ipart(), Iavg(), partId, frId, mvId, true);
				//TODO check how we have to average the movie particles to calculate the correlations

				//To obtain the maximum value of correlation we need the "particle", all frames averaged and stored in Imean
				Image<double> Imean;
				Imean().initZeros(XSIZE(Iproj()),YSIZE(Iproj()));
				averagingAll(mdPart, Ipart(), Imean(), partId, frId, mvId, false);
				double meanD, stdD;
				Imean().computeAvgStdev(meanD,stdD);

				//To study what happens with the ML with a Poisson
				std::cerr << "FOR Movie ID: " << mvId << ". Working with Slope: " << slope << ". Intercept: " << intercept <<  std::endl;
				double likelihoodRef=0.;
				double likelihoodFr=0.;
				double likelihoodAvg=0.;
				double lambda, factRef, factFr, factAvg;
				double maxShift = XSIZE(Imean())/2-10;
				double maxShift2 = maxShift*maxShift;
				int count=0;
				for(int n=0; n<YSIZE(Imean()); n++){
					for(int m=0; m<XSIZE(Imean()); m++){
						if ((n-round(YSIZE(Imean())/2))*(n-round(YSIZE(Imean())/2))+(m-round(XSIZE(Imean())/2))*(m-round(XSIZE(Imean())/2))>maxShift2) // continue if the Euclidean distance is too far
							continue;
						count++;
						factRef=1.;
						lambda = slope*DIRECT_A2D_ELEM(projV(), n, m)+intercept;
						if (DIRECT_A2D_ELEM(Imean(), n, m)>0){
							for(int aa=1; aa<=DIRECT_A2D_ELEM(Imean(), n, m); aa++)
								factRef = factRef*aa;
						}
						likelihoodRef += -1.*lambda + DIRECT_A2D_ELEM(Imean(), n, m)*log(lambda) - log(factRef);

						factFr=1.;
						if (DIRECT_A2D_ELEM(Ipart(), n, m)>0){
							for(int aa=1; aa<=DIRECT_A2D_ELEM(Ipart(), n, m); aa++)
								factFr = factFr*aa;
						}
						likelihoodFr += -1.*lambda + DIRECT_A2D_ELEM(Ipart(), n, m)*log(lambda) - log(factFr);

						factAvg=1.;
						if (DIRECT_A2D_ELEM(Iavg(), n, m)>0){
							for(int aa=1; aa<=DIRECT_A2D_ELEM(Iavg(), n, m); aa++)
								factAvg = factAvg*aa;
						}
						likelihoodAvg += -1.*lambda + DIRECT_A2D_ELEM(Iavg(), n, m)*log(lambda) - log(factAvg);
					}
				}

				/*/AJ para leer phantom y promediar
				Image<double> Itot, Imean;
				Iproj.read("projNueva.mrc");
				Iproj().setXmippOrigin();
				Itot().initZeros(XSIZE(Iproj()),YSIZE(Iproj()));
				Imean().initZeros(XSIZE(Iproj()),YSIZE(Iproj()));
				//if (partId==5818){
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

					double Dmin, Dmax, irange, val;
					projV().computeDoubleMinMax(Dmin, Dmax);
					irange=1.0/(Dmax - Dmin);
					FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(projV()){
						val=DIRECT_MULTIDIM_ELEM(projV(),n);
						DIRECT_MULTIDIM_ELEM(projV(),n) = (Dmax - val) * irange;
					}
					Iavg.write(formatString("myAvg.mrc"));
					projV.write(formatString("myProj.mrc"));

				//FIN AJ phantom lectura/*/

			//cutfreq = inifreq + step*n;


			//filtering the projected particles with the lowpass filter
			FilterLP.w1=(n+1)*bandSize;
			//Filter.w1=n*bandSize;
			//Filter.w2=(n+1)*bandSize;
			//printf("Filter data %lf %lf %lf \n", bandSize, n*bandSize, (n+1)*bandSize);
			FilterLP.generateMask(projV());
			FilterLP.applyMaskSpace(projV());

			//filtering the averaged movie particle (leaving out the j-frame)
			FilterLP.generateMask(Iavg());
			FilterLP.applyMaskSpace(Iavg());

			//filtering the averaged movie particle (all the frames)
			FilterLP.generateMask(Imean());
			FilterLP.applyMaskSpace(Imean());

			//Iavg.write(formatString("myAvgFiltered.mrc"));
			//projV.write(formatString("myProjFiltered.mrc"));


			//calculate similarity measures between averaged movie particles and filtered projection
			double weight;
			double corrNRef, corrMRef, corrWRef, imedRef;
			similarity(projV(), Imean(), corrNRef, corrMRef, corrWRef, imedRef);
			//printf("REF CORRELATION %8.20lf \n", corrNRef);
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

			//weight = corrNRef-corrN; //TODO
			weight = corrN; //TODO
			printf("%lf %lf  --  %lf %lf %lf \n", corrNRef, corrN, likelihoodRef, likelihoodFr, likelihoodAvg);
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
			totalWeight[n]+=weight;

				//}

		} //end frequencies loop


		if (numPartPerMov[countMv]==partCount && frId==nFrames){
			//std::cerr << " maxvalue " << maxvalue << std::endl;
			//std::cerr << "  matrixWeights " << matrixWeights << std::endl;
			//std::cerr << " matrixWeights " << matrixWeights << std::endl;
			//matrixWeights/=maxvalue;

			double tot;
			std::cerr << "- ANTES matrixWeights " << matrixWeights << std::endl;
			for(int n=0; n<Nsteps; n++){
				tot=-1.;
				for(int mm=0; mm<nFrames; mm++){
					DIRECT_ZYX_ELEM(matrixWeights, partCount-1, mm, n) = totalWeight[n]-DIRECT_ZYX_ELEM(matrixWeights, partCount-1, mm, n);
					if (DIRECT_ZYX_ELEM(matrixWeights, partCount-1, mm, n)>tot)
						tot=DIRECT_ZYX_ELEM(matrixWeights, partCount-1, mm, n);
				}
				for(int mm=0; mm<nFrames; mm++){
					DIRECT_ZYX_ELEM(matrixWeights, partCount-1, mm, n) /= tot;
				}
			}

			std::cerr << "- NORM matrixWeights " << matrixWeights << std::endl;
			//std::cerr << "X " << XSIZE(matrixWeights) << " Y " << YSIZE(matrixWeights) << " Z " << ZSIZE(matrixWeights) << std::endl;
			for (int s=0; s<ZSIZE(matrixWeights); s++){
				Matrix2D<double> matrixWeightsMat;
				MultidimArray<double> matrixWeightsSlice;
				matrixWeights.getSlice(s, matrixWeightsSlice);
				//std::cerr << "- Slice: " << XSIZE(matrixWeightsSlice) << " " << YSIZE(matrixWeightsSlice) << " " << ZSIZE(matrixWeightsSlice) << std::endl;
				matrixWeightsSlice.copy(matrixWeightsMat);
				//std::cerr << "- Matrix: " << matrixWeightsMat.Xdim() << " " << matrixWeightsMat.Ydim() << " " << matrixWeightsMat << std::endl;
				Matrix2D<double> U,V;
				Matrix1D<double> S;
				svdcmp(matrixWeightsMat,U,S,V);
				//std::cerr << "- SVD: " << U << std::endl;
				//AJ prueba
				Matrix2D<double> Smat;
				Smat.initZeros(S.size(),S.size());
				for (int h=0; h<S.size(); h++){
					if (h<2)
						Smat(h,h)=S(h);
				}
				Matrix2D<double> result1;
				matrixOperation_AB(U, Smat, result1);
				matrixOperation_ABt(result1, V, resultWeights);
				std::cerr << "- SVD recons: " << resultWeights << std::endl;

				//We must do something with the previously obtained values
				FileName fnTest;
				Ifinal().resize(projV());
				Ifinal().initZeros();
				IfinalAux().resize(projV());
				IfinalAux().initZeros();
				double shiftFrameX, shiftFrameY;
				for(int mm=0; mm<nFrames; mm++){
					IfinalAux().initZeros();
					mdPart.getRow(currentRow2, iterPart2->objId);
					currentRow2.getValue(MDL_IMAGE,fnTest);
					currentRow2.getValue(MDL_POLISHING_X,shiftFrameX);
					currentRow2.getValue(MDL_POLISHING_Y,shiftFrameY);
					std::vector<double> coeffsToStore;

					for(int nn=0; nn<Nsteps; nn++){
						IpartOut.read(fnTest);
						IpartOut().setXmippOrigin();
						FilterBP.w1=nn*bandSize;
						FilterBP.w2=(nn+1)*bandSize;
						//TODO ask this Low pass or Band pass?
						//FilterLP.w1=(nn+1)*bandSize;
						FilterBP.generateMask(IpartOut());
						FilterBP.applyMaskSpace(IpartOut());

						//printf("PESO APLICADO %lf, para frame %d y frecuencia %d \n", resultWeights(mm, nn), mm, nn);
						IfinalAux()+=IpartOut()*resultWeights(mm, nn);
						coeffsToStore.push_back(resultWeights(mm, nn));

					}
					//IfinalAux().setXmippOrigin();
					Ifinal()+=IfinalAux();

					if(iterPart2->hasNext())
						iterPart2->moveNext();
				}
				Ifinal.write(fnStackOut2,countOutMd,true,WRITE_APPEND);
				countOutMd++;
				FileName fnToSave2;
				fnToSave2.compose(countOutMd, fnStackOut2);
				//size_t id = SFq.addObject();
				//SFq.setValue(MDL_IMAGE, fnToSave, id);
				currentRow.setValue(MDL_IMAGE, fnToSave2);
				SFq2.addRow(currentRow);

			}

		}

		if(iterPart->hasNext())
			iterPart->moveNext();

	} //end movie particles loop

	FileName fnOut2;
	fnOut2 = fnMdMov.insertBeforeExtension("_out_filtered");
	printf("Writing output metadata 2 \n");
	printf("%s \n ", fnOut2.getString().c_str());
	SFq2.write(fnOut2);


	//MultidimArray<double> weightsperfreq;
	//weightsperfreq.initZeros(NSIZE(matrixWeights), ZSIZE(matrixWeights), YSIZE(matrixWeights), XSIZE(matrixWeights));
	//calculateFrameWeightPerFreq(matrixWeights, weightsperfreq, maxvalues);

	delete[] ctfs;

}



