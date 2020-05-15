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
#include <core/xmipp_fft.h>


void ProgParticlePolishing::defineParams()
{
    addUsageLine("Particle polishing from a stack of movie particles");
    addParamsLine(" -i <movie>: Input movie particle metadata");
    //addParamsLine(" -iPart <part>: Input particles metadata");
    addParamsLine(" -vol <volume>: Input volume to generate the reference projections");
    addParamsLine(" --s <samplingRate=1>: Sampling rate");
    addParamsLine(" --nFrames <nFrames>: Number of frames");
    addParamsLine(" --nMics <nMics>: Number of micrographs");
    addParamsLine(" --filter <nFilter=1>: The number of filters to apply");
    addParamsLine(" --movxdim <xmov> : Movie size in x dimension");
    addParamsLine(" --movydim <ymov> : Movie size in y dimension");
    addParamsLine(" [-o <fnOut=\"out.xmd\">]: Output metadata with weighted particles");
    addParamsLine(" [--fixedBW]      : fixed bandwith for the filters. If this flag does not appear, the bandwith will be lower in low frequencies");

}

void ProgParticlePolishing::readParams()
{
	fnMdMov=getParam("-i");
	//fnMdPart=getParam("-iPart");
	fnVol = getParam("-vol");
	fnOut=getParam("-o");
	nFrames=getIntParam("--nFrames");
	nMics=getIntParam("--nMics");
	nFilters=getIntParam("--filter");
	samplingRate=getDoubleParam("--s");
	xmov = getIntParam("--movxdim");
	ymov = getIntParam("--movydim");
    fixedBW  = checkParam("--fixedBW");

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


void ProgParticlePolishing::produceSideInfo()
{

	//MOVIE PARTICLES IMAGES
	MetaData mdPartPrev, mdPart;
	MDRow currentRow;
	mdPartPrev.read(fnMdMov,NULL);
	mdPartSize = mdPartPrev.size();
	mdPart.sort(mdPartPrev, MDL_PARTICLE_ID); //or sort by MDL_MICROGRAPH_ID ¿?

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

void ProgParticlePolishing::averagingAll(const MetaData &mdPart, const MultidimArray<double> &I, MultidimArray<double> &Iout, size_t partId, size_t frameId, size_t movieId, bool noCurrent, bool applyAlign){

	//MDRow currentRow;
	FileName fnPart;
	Image<double> Ipart;
	size_t newPartId, newFrameId, newMovieId;
	//size_t mdPartSize = mdPart.size();
	double count;
	std::vector<int>::iterator it2;

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
				if(applyAlign){
					it2 = find(partIds.begin(), partIds.end(), (int)partId);
					int index = std::distance(partIds.begin(), it2);
					selfTranslate(NEAREST, Ipart(), vectorR2((double)resultShiftX[index+newFrameId-1], (double)resultShiftY[index+newFrameId-1]), DONT_WRAP, 0.0);
				}
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


void ProgParticlePolishing::writingOutput(size_t xdim, size_t ydim){

	//MOVIE PARTICLES IMAGES
	MetaData mdPartPrev, mdPart;
	MDRow currentRow;
	mdPartPrev.read(fnMdMov,NULL);
	mdPartSize = mdPartPrev.size();
	mdPart.sort(mdPartPrev, MDL_PARTICLE_ID);

	MDIterator *iterPart = new MDIterator();
	std::vector<int>::iterator it2;

	double rot, tilt, psi, x, y;
	bool flip;
	size_t frId, mvId, partId;
	int xcoor, ycoor;
	int enabled;
	FileName fnPart;
	String aux;

	MetaData SFq2;
	FileName fnRoot2=fnMdMov.insertBeforeExtension("_out_particles_new");
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
			if(iterPart->hasNext())
				iterPart->moveNext();
			continue;
		}

		if(partIdPrev!=partId){
			countForPart=0;
			Ifinal().initZeros(ydim, xdim);
			Ifinal().setXmippOrigin();
			partIdPrev=partId;
		}
		currentRow.getValue(MDL_IMAGE,fnPart);
		Ipart.read(fnPart);
		Ipart().setXmippOrigin();

		it2 = find(partIds.begin(), partIds.end(), (int)partId);
		int index = std::distance(partIds.begin(), it2);
		currentRow.getValue(MDL_FRAME_ID,frId);
		//printf("Particle %d frame %d shiftX %d shiftY %d \n", partId, frId, resultShiftX[index+countForPart], resultShiftY[index+countForPart]);
		selfTranslate(NEAREST, Ipart(), vectorR2((double)resultShiftX[index+frId-1], (double)resultShiftY[index+frId-1]), DONT_WRAP, 0.0);
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

		if(iterPart->hasNext())
			iterPart->moveNext();

	} //end mdPartSize loop

	FileName fnOut2;
	fnOut2 = fnMdMov.insertBeforeExtension("_out_new");
	printf("Writing output metadata\n");
	printf("%s \n ", fnOut2.getString().c_str());
	SFq2.write(fnOut2);

}

void ProgParticlePolishing::calculateWeightedFrequency(MultidimArray<double> &Ipart, int Nsteps, const double *frC, const std::vector<double> weights){

	MultidimArray< std::complex<double> > fftIpart;
	MultidimArray<double> wfftIpart;
	FourierTransformer transformer;
	// Auxiliary vector for representing frequency values
	Matrix1D<double> fidx;

	transformer.FourierTransform(Ipart, fftIpart, false);

    //MultidimArray<double> vMag;
    //FFT_magnitude(fftIpart, vMag);
	//Image<double> imageA;
	//imageA()=vMag;
    //imageA.write(formatString("amplitude1.tif"));

	//wfftIpart.initZeros(fftIpart);
	fidx.resizeNoCopy(3);
    for (size_t k=0; k<ZSIZE(fftIpart); k++)
    {
        FFT_IDX2DIGFREQ(k,ZSIZE(Ipart),ZZ(fidx));
        for (size_t i=0; i<YSIZE(fftIpart); i++)
        {
            FFT_IDX2DIGFREQ(i,YSIZE(Ipart),YY(fidx));
            for (size_t j=0; j<XSIZE(fftIpart); j++)
            {
                FFT_IDX2DIGFREQ(j,XSIZE(Ipart),XX(fidx));
                double absw = fidx.module();
                for (int n=0; n<Nsteps; n++){
                	if(absw<=frC[n] && n==0){
                    	//DIRECT_A3D_ELEM(wfftIpart,k,i,j)=weights[0];
                		DIRECT_A3D_ELEM(fftIpart,k,i,j)*= weights[0];
                    	break;
                	}else if((absw<=frC[n] && n!=0)){
                		//DIRECT_A3D_ELEM(wfftIpart,k,i,j)=((weights[n]-weights[n-1])/((n+1)*bandSize - (n)*bandSize)*(absw - (n)*bandSize)) + weights[n-1];
                		DIRECT_A3D_ELEM(fftIpart,k,i,j)*= (((weights[n]-weights[n-1])/(frC[n] - frC[n-1]))*(absw - frC[n-1])) + weights[n-1];
                		break;
                	}else if((absw>=frC[n] && n==Nsteps-1)){
                    	//DIRECT_A3D_ELEM(wfftIpart,k,i,j)=weights[Nsteps-1];
                		DIRECT_A3D_ELEM(fftIpart,k,i,j)*= weights[Nsteps-1];
                    	break;
                	}
                }
            }
        }
    }
	//Image<double> imageW;
	//imageW()=wfftIpart;
    //imageW.write(formatString("pesos.tif"));

    transformer.inverseFourierTransform();

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
	Image<double> Ipart, projV, Iavg, Iproj, IpartOut, Iout, Ifinal, IfinalAux, projAux;
	Image<double> Ipartaux, projVaux;
	//MDRow currentRow;
	//MDRow currentRow2;
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
	resultShiftX = new int[partIds.size()];
	resultShiftY = new int[partIds.size()];

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

		dataInMovie = 0;
		vectorAvg.initZeros(2, nStep);
		slope=0.;
		intercept=0.;

		for(int i=0; i<mdPartSize; i++){

			frId = frIds[i];
			mvId = mvIds[i];
			partId = partIds[i];
			enabled = enables[i];

			if(enabled==-1){
				partIdDone.push_back((int)partId);
				continue;
			}

			if(mvId!=m){
				continue;
			}

			it = find(partIdDone.begin(), partIdDone.end(), (int)partId);
			if (it != partIdDone.end()){
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
			bool averageAll=false;
			bool applyAlign=false;
			averagingAll(mdPart, Ipart(), Iavg(), partId, frId, mvId, averageAll, applyAlign);

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

	/*MetaData SFq;
	FileName fnRoot=fnMdMov.insertBeforeExtension("_out_particles");
	FileName fnStackOut=formatString("%s.stk",fnRoot.c_str());
	if(fnStackOut.exists())
		fnStackOut.deleteFile();*/


	//iterPart->init(mdPart);
	//iterPart2->init(mdPart);
	int countForPart=0;
	int countDisabled=0;
	size_t frId, mvId, partId;
	int enabled;

	for(int ii=0; ii<mdPartSize; ii++){

		//Project the volume with the parameters in the image
		double rot, tilt, psi, xValue, yValue;
		bool flip;
		int xcoor, ycoor;

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
			continue;
		}


		it = find(partIdDone2.begin(), partIdDone2.end(), (int)partId);
		if (it != partIdDone2.end()){
			if (countDisabled==1){
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

		projectVolume(*projectorV, PV, xdim, xdim,  rot, tilt, psi);
		applyGeometry(LINEAR,projVaux(),PV(),A,IS_INV,DONT_WRAP,0.);
		projVaux().setXmippOrigin();

		lkresults.initZeros(myL, myL);
		for(int jj=0; jj<myL; jj++){
			for(int hh=0; hh<myL; hh++){

				projV().initZeros(projVaux());
				projV().setXmippOrigin();

				if (shiftX[jj]!=0 || shiftY[hh]!=0)
					translate(LINEAR, projV(), projVaux(), vectorR2((double)shiftX[jj], (double)shiftY[hh]), DONT_WRAP, 0.0); //TODO: projV o Ipart deben moverse¿? projV para evitar problemas de interpolacion ¿?
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

		//negative because these values are calculated with a displacement in the projection, but then they will be applied to the frames
		finalPosX[countForPart]=-shiftX[bestPosX];
		finalPosY[countForPart]=-shiftY[bestPosY];

		printf(". BEST POS for particle %d. Shift %lf, %lf \n", countForPart, finalPosX[countForPart], finalPosY[countForPart]);

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
			std::vector<int>::iterator it2;
			it2 = find(partIds.begin(), partIds.end(), (int)partId);
			int index = std::distance(partIds.begin(), it2);
			//printf("INDEX %d \n", index);

			for(int mm=0; mm<nFrames; mm++){

				int pos = index+mm;
				resultShiftX[pos]=(int)round(resultX(mm));
				resultShiftY[pos]=(int)round(resultY(mm));
				printf("Particle %d with id %d in frame %d: shiftX %lf shiftY %lf \n", ii, (int)partId, mm, resultX(mm), resultY(mm));
				/*if(resultShiftX.size() != (ii-(nFrames-1)+mm))
					std::cerr << "Error saving data" << std::endl;
				else{
					resultShiftX.push_back(resultX(mm));
					resultShiftY.push_back(resultY(mm));
					printf("Particle %d frame %d shiftX %lf shiftY %lf \n", ii, mm, resultX(mm), resultY(mm));
				}*/
			}
			/*std::cout << "RESULTS" << std::endl;
			for(int mm=0; mm<partIds.size(); mm++)
				printf("%d ",resultShiftX[mm]);
			printf("\n");
			for(int mm=0; mm<partIds.size(); mm++)
				printf("%d ",resultShiftY[mm]);
			printf("\n");*/

		}

		//if(iterPart->hasNext())
			//iterPart->moveNext();

	}//end loop particles


	/*FileName fnOut;
	fnOut = fnMdMov.insertBeforeExtension("_out");
	printf("Writing output metadata \n");
	printf("%s \n ", fnOut.getString().c_str());
	SFq.write(fnOut);*/

	writingOutput(XSIZE(Ipart()), YSIZE(Ipart()));

	//exit(0);

	/////////////////////////
	//Second Part
	/////////////////////////

	MetaData mdPartPrev, mdPart;
	MDRow currentRow;
	mdPartPrev.read(fnMdMov,NULL);
	mdPart.sort(mdPartPrev, MDL_PARTICLE_ID);

	MultidimArray<double> matrixWeights, maxvalues, matrixWeightsPart;
	FourierFilter FilterBP, FilterLP;
	FilterBP.FilterBand=BANDPASS;
	FilterBP.FilterShape=RAISED_COSINE;
	FilterLP.FilterBand=LOWPASS;
	FilterLP.FilterShape=RAISED_COSINE;
	double cutfreq;
	//double inifreq=0.25; //0.05;
	//double step=0.25; //0.05;
	//int Nsteps= int((0.5-inifreq)/step);
	double bandSize=0.5/(double)nFilters;
	double frC[nFilters];

	if(fixedBW){
		for (int n=0; n<nFilters; n++)
			frC[n]=(n+1)*bandSize;
	}else{
		for (int n=nFilters; n>0; n--){
			if(n==nFilters)
				frC[n-1]=0.5;
			else
				frC[n-1]=frC[n]/2;
		}
	}

	printf("FREQ: \n");
	for (int n=0; n<nFilters; n++)
		printf(" %lf ",frC[n]);
	printf("\n");

	//double maxvalue=-1.;

	/*std::vector<int> numPartPerMov;
	for(int i=0; i<=nMics; i++){
		numPartPerMov.push_back(0);
	}
	int prevId=-1;
	for(int i=0; i<mvIds.size(); i++){
		if(enables[i]==1 && partIds[i]!=prevId){
			numPartPerMov[mvIds[i]]++;
			prevId = partIds[i];
		}
	}
	//std::cerr << " size numPartPerMov " << numPartPerMov.size() << std::endl;
	//for (int i=0; i<numPartPerMov.size(); i++)
	//	std::cerr << " numPartPerMov[" << i << "] = "<< numPartPerMov[i] << std::endl;
	*/

	//exit(0);

	int prevMvId=-1, prevPartId=-1, partCount=0;
	int countOutMd=0;
	prevMvId=-1;
	prevPartId=-1;
	partCount=0;
	int countMv=-1;
	iterPart->init(mdPart);
	//iterPart2->init(mdPart);

	Matrix2D<double> resultWeights;
	double totalWeight[nFilters];

	MetaData SFq2;
	FileName fnRoot2=fnMdMov.insertBeforeExtension("_out_particles_filtered_new");
	FileName fnStackOut2=formatString("%s.stk",fnRoot2.c_str());
	if(fnStackOut2.exists())
		fnStackOut2.deleteFile();


	std::vector<int>::iterator it2;
	countForPart=0;
	for(int i=0; i<mdPartSize; i++){
		
		//Project the volume with the parameters in the image
		double rot, tilt, psi, x, y;
		bool flip;
		size_t frId, mvId, partId;
		int xcoor, ycoor;
		int enabled;

		frId = frIds[i];
		mvId = mvIds[i];
		partId = partIds[i];
		enabled = enables[i];

		mdPart.getRow(currentRow, iterPart->objId);
		/*currentRow.getValue(MDL_IMAGE,fnPart);
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
		currentRow.getValue(MDL_ENABLED,enabled);*/


		if(enabled==-1){
			if(iterPart->hasNext())
				iterPart->moveNext();
			/*if(iterPart2->hasNext())
				iterPart2->moveNext();*/
			continue;
		}

		if(prevMvId!=mvId){
			countMv++;
			partCount=0;
			prevMvId=mvId;
			//maxvalue=-1.;
			//matrixWeights.initZeros(numPartPerMov[mvId], nFrames, nFilters);
		}
		if(partId!=prevPartId){
			countForPart=0;
			matrixWeightsPart.initZeros(nFrames, nFilters);
			prevPartId=partId;
			partCount++;
			//for(int n=0; n>nFilters; n++)
			//	totalWeight[n]=0.;
		}else{
			countForPart++;
		}

		fnPart = (String)fnParts[i];
		//Ipart.read(fnPart);
		//Ipart().setXmippOrigin();
		rot = rots[i];
		tilt=tilts[i];
		psi=psis[i];
		x = xs[i];
		y = ys[i];
		flip = flips[i];
		xcoor = xcoors[i];
		ycoor = ycoors[i];

		it2 = find(partIds.begin(), partIds.end(), (int)partId);
		int index = std::distance(partIds.begin(), it2);

		A.initIdentity(3);
		MAT_ELEM(A,0,2)=x;
		MAT_ELEM(A,1,2)=y;
		if (flip){
			MAT_ELEM(A,0,0)*=-1;
			MAT_ELEM(A,0,1)*=-1;
			MAT_ELEM(A,0,2)*=-1;
		}

		//Reading the frame
		Ipartaux.read(fnPart);
		Ipartaux().setXmippOrigin();

		//Creating the projection image
		projectVolume(*projectorV, PV, xdim, xdim,  rot, tilt, psi);
		applyGeometry(LINEAR,projVaux(),PV(),A,IS_INV,DONT_WRAP,0.);
		projVaux().setXmippOrigin();

		//To invert contrast in the projections
		double Dmin, Dmax, irange, val;
		projVaux().computeDoubleMinMax(Dmin, Dmax);
		irange=1.0/(Dmax - Dmin);

		//filtering the projections with the ctf
		ctf = ctfs[i];
		ctf.produceSideInfo();


		for(int n=0; n<nFilters; n++){

			projV().initZeros(projVaux());
			projV().setXmippOrigin();
			//To invert contrast in the projections
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(projVaux()){
				val=DIRECT_MULTIDIM_ELEM(projVaux(),n);
				DIRECT_MULTIDIM_ELEM(projV(),n) = (Dmax - val) * irange;
			}
			//filtering the projections with the ctf
			ctf.applyCTF(projV(), samplingRate, false);


			//Applying the displacement previously obtained to the frame
			Ipart().initZeros(Ipartaux());
			Ipart().setXmippOrigin();
			translate(NEAREST, Ipart(), Ipartaux(), vectorR2((double)resultShiftX[index+frId-1], (double)resultShiftY[index+frId-1]), DONT_WRAP, 0.0);

			//averaging movie particle image with the ones in all the frames but without the current one (the first true), and applying the align to all of them (last true)
			bool averageAll=true;
			bool applyAlign=true;
			averagingAll(mdPart, Ipart(), Iavg(), partId, frId, mvId, averageAll, applyAlign);
			//TODO check how we have to average the movie particles to calculate the correlations

			//cutfreq = inifreq + step*n;


			//filtering the projected particles with the lowpass filter
			FilterLP.w1=frC[n]; //(n+1)*bandSize;
			//Filter.w1=n*bandSize;
			//Filter.w2=(n+1)*bandSize;
			//printf("Filter data %lf %lf %lf \n", bandSize, n*bandSize, (n+1)*bandSize);
			FilterLP.generateMask(projV());
			FilterLP.applyMaskSpace(projV());

			//filtering the averaged movie particle (leaving out the j-frame)
			FilterLP.generateMask(Iavg());
			FilterLP.applyMaskSpace(Iavg());

			//filtering the averaged movie particle (all the frames)
			//FilterLP.generateMask(Imean());
			//FilterLP.applyMaskSpace(Imean());

			//Iavg.write(formatString("myAvgFiltered.mrc"));
			//projV.write(formatString("myProjFiltered.mrc"));


			//calculate similarity measures between averaged movie particles and filtered projection
			double weight;
			//double corrNRef, corrMRef, corrWRef, imedRef;
			//similarity(projV(), Imean(), corrNRef, corrMRef, corrWRef, imedRef);
			//printf("REF CORRELATION %8.20lf \n", corrNRef);
			double corrN, corrM, corrW, imed;
			similarity(projV(), Iavg(), corrN, corrM, corrW, imed); //with or without meanD (as last parameter) ¿?
			//TODO: la corrW que viene de Idiff no tiene mucho sentido porque la Idiff en este caso no nos dice nada


			weight = corrN; //TODO
			//printf("%lf %lf  \n", corrNRef, corrN);
			//if(weight>maxvalue)
			//	maxvalue=weight;

			DIRECT_A2D_ELEM(matrixWeightsPart, frId-1, n) = weight;
			//DIRECT_ZYX_ELEM(matrixWeights, partCount-1, frId-1, n) = weight;
			//totalWeight[n]+=weight;


		} //end frequencies loop


		//if (numPartPerMov[mvId]==partCount && frId==nFrames){
		if (countForPart==nFrames-1){

			//std::cerr << "- ANTES matrixWeightsPart " << matrixWeightsPart << std::endl;
			//to normalize the weights
			for(int n=0; n<nFilters; n++){
				double aux=0, aux2=0;
				for(int mm=0; mm<nFrames; mm++){
					aux += DIRECT_A2D_ELEM(matrixWeightsPart, mm, n);
				}
				//std::cerr << "1 Values " << aux << std::endl;
				for(int mm=0; mm<nFrames; mm++){
					DIRECT_A2D_ELEM(matrixWeightsPart, mm, n) = (2*(aux/nFrames) - DIRECT_A2D_ELEM(matrixWeightsPart, mm, n)); //aux;
					if (DIRECT_A2D_ELEM(matrixWeightsPart, mm, n)<0)
						DIRECT_A2D_ELEM(matrixWeightsPart, mm, n)=0.;
					aux2 += DIRECT_A2D_ELEM(matrixWeightsPart, mm, n);
				}
				//std::cerr << "2 Values " << aux2 << std::endl;
				for(int mm=0; mm<nFrames; mm++){
					if (aux2!=0)
						DIRECT_A2D_ELEM(matrixWeightsPart, mm, n) /= aux2;
					else
						DIRECT_A2D_ELEM(matrixWeightsPart, mm, n) = 0.;
				}
			}
			//std::cerr << "- NORM matrixWeights " << matrixWeightsPart << std::endl;

			Matrix2D<double> matrixWeightsMat;
			matrixWeightsPart.copy(matrixWeightsMat);
			//std::cerr << "- Matrix: " << matrixWeightsMat.Xdim() << " " << matrixWeightsMat.Ydim() << " " << matrixWeightsMat << std::endl;
			Matrix2D<double> U,V;
			Matrix1D<double> S;
			svdcmp(matrixWeightsMat,U,S,V);
			//std::cerr << "- SVD: " << U << std::endl;
			//AJ testing
			Matrix2D<double> Smat;
			Smat.initZeros(S.size(),S.size());
			for (int h=0; h<S.size(); h++){
				if (h<2)
					Smat(h,h)=S(h);
			}
			Matrix2D<double> result1;
			matrixOperation_AB(U, Smat, result1);
			matrixOperation_ABt(result1, V, resultWeights);
			std::cerr << "For particle " << fnParts[i] << std::endl;
			std::cerr << "- SVD recons: " << resultWeights << std::endl;


			//We must do something with the previously obtained values
			FileName fnTest;
			Ifinal().initZeros(projV());
			Ifinal().setXmippOrigin();
			double shiftFrameX, shiftFrameY;
			std::vector<double> myweights;

			it2 = find(partIds.begin(), partIds.end(), (int)partId);
			int index = std::distance(partIds.begin(), it2);
			for(int mm=0; mm<nFrames; mm++){

				fnPart = (String)fnParts[i-(nFrames-1-mm)];
				Ipartaux.read(fnPart);
				Ipartaux().setXmippOrigin();

				selfTranslate(NEAREST, Ipartaux(), vectorR2((double)resultShiftX[index+mm-1], (double)resultShiftY[index+mm-1]), DONT_WRAP, 0.0);

				for(int nn=0; nn<nFilters; nn++){

					/*IfinalAux().initZeros(projV());
					IfinalAux().setXmippOrigin();
					translate(NEAREST, IfinalAux(), Ipartaux(), vectorR2((double)resultShiftX[index+mm-1], (double)resultShiftY[index+mm-1]), DONT_WRAP, 0.0);

					FilterBP.w1=nn*bandSize;
					FilterBP.w2=(nn+1)*bandSize;
					//TODO ask this Low pass or Band pass?
					//FilterLP.w1=(nn+1)*bandSize;
					FilterBP.generateMask(IfinalAux());
					FilterBP.applyMaskSpace(IfinalAux());

					//printf("%s PESO APLICADO %lf, para frame %d y frecuencia %d \n", fnPart.c_str(), resultWeights(mm, nn), mm, nn);
					Ifinal()+=IfinalAux()*resultWeights(mm, nn);
					*/
					myweights.push_back(resultWeights(mm, nn)); //resultWeights(mm, nn)

				}

				calculateWeightedFrequency(Ipartaux(), nFilters, frC, myweights);
				//Ifinal.write(formatString("unFrameDespuesFiltro.tif"));
				//exit(0);
				Ifinal()+=Ipartaux();

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

		if(iterPart->hasNext())
			iterPart->moveNext();

	} //end movie particles loop

	FileName fnOut2;
	fnOut2 = fnMdMov.insertBeforeExtension("_out_filtered_new");
	printf("Writing output metadata 2 \n");
	printf("%s \n ", fnOut2.getString().c_str());
	SFq2.write(fnOut2);

	delete[] ctfs;

}



