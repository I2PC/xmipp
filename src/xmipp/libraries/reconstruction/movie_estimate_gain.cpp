/***************************************************************************
 *
 * Authors:        Victoria Peredo
 *                 Estrella Fernandez
 *                 Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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

#include "movie_estimate_gain.h"

void ProgMovieEstimateGain::defineParams()
{
    addUsageLine("Estimate the gain image for a movie");
    addParamsLine(" -i <movie>: Input movie");
    addParamsLine(" [--oroot <fn=\"estimated\">]: Estimated corrections and gains");
    addParamsLine("                             :+(Ideal=Observed*Corr, Observed=Ideal*Gain)");
    addParamsLine(" [--iter <N=3>]: Number of iterations");
    addParamsLine(" [--sigma <s=-1>]: Sigma to use, if negative, then it is searched");
    addParamsLine(" [--maxSigma <s=3>]: Maximum number of neighbour rows/columns to analyze");
    addParamsLine(" [--frameStep <s=1>]: Skip frames during the estimation");
    addParamsLine("                    : Set to 1 to process all frames, set to 2 to process 1 every 2 frames, ...");
    addParamsLine(" [--sigmaStep <s=0.5>]: Step size for sigma");
    addParamsLine(" [--singleRef] : Use a single histogram reference");
    addParamsLine("               :+This assumes that there is no image contamination or carbon holes");
    addParamsLine(" [--gainImage <fn=\"\">] : Reference to external gain image (we will multiply by it)");
    addParamsLine(" [--applyGain <fnOut>] : Flag for using external gain image");
    addParamsLine("               : applyGain=True will use external gain image");
}

void ProgMovieEstimateGain::readParams()
{
	fnIn=getParam("-i");
	fnRoot=getParam("--oroot");
	Niter=getIntParam("--iter");
	sigma=getIntParam("--sigma");
	maxSigma=getDoubleParam("--maxSigma");
	sigmaStep=getDoubleParam("--sigmaStep");
	singleReference=checkParam("--singleRef");
	frameStep=getIntParam("--frameStep");
	fnGain=getParam("--gainImage");
	applyGain=checkParam("--applyGain");
    if (applyGain)
       fnCorrected=getParam("--applyGain");
}

void ProgMovieEstimateGain::produceSideInfo()
{
	if (fnIn.getExtension()=="mrc")
		fnIn+=":mrcs";
	mdIn.read(fnIn);
	mdIn.removeDisabled();
	if (mdIn.size()==0)
		exit(0);
	Image<double> Iframe;
	FileName fnFrame;
	mdIn.getValue(MDL_IMAGE,fnFrame,mdIn.firstObject());
	Iframe.read(fnFrame);
	Xdim=XSIZE(Iframe());
	Ydim=YSIZE(Iframe());

	columnH.initZeros(Ydim,Xdim);
	rowH.initZeros(Ydim,Xdim);
	if (fnGain!="")
	{
		ICorrection.read(fnGain);
		if (YSIZE(ICorrection())!=Ydim || XSIZE(ICorrection())!=Xdim)
			REPORT_ERROR(ERR_MULTIDIM_SIZE,"The gain image and the movie do not have the same dimensions");
	}
	else
	{
		ICorrection().resizeNoCopy(Ydim,Xdim);
		ICorrection().initConstant(1);
	}

	sumObs.initZeros(Ydim,Xdim);

    int iFrame=0;
	FOR_ALL_OBJECTS_IN_METADATA(mdIn)
	{
	    if (iFrame%frameStep==0)
	    {
               mdIn.getValue(MDL_IMAGE,fnFrame,__iter.objId);
               Iframe.read(fnFrame);
               MultidimArray<double>  mIframe=Iframe();
               MultidimArray<double>  mICorrection=ICorrection();
               FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mIframe)
                   DIRECT_MULTIDIM_ELEM(mIframe,n) /= DIRECT_MULTIDIM_ELEM(mICorrection,n);               sumObs+=Iframe();
	    }
	    iFrame++;
	}
	sumObs*=2;

	// Initialize sigma values
	for (double sigma=0; sigma<=maxSigma; sigma+=sigmaStep)
		listOfSigmas.push_back(sigma);

	for (size_t i=0; i<listOfSigmas.size(); ++i)
	{
	    int jmax=ceil(3*listOfSigmas[i]);
	    listOfWidths.push_back(jmax);
	    double *weights=new double[jmax+1];
	    double K=1;
            if (listOfSigmas[i]>0)
               K=-0.5/(listOfSigmas[i]*listOfSigmas[i]);
	    for (int j=0; j<=jmax; ++j)
	       weights[j]=exp(K*j*j);
	    listOfWeights.push_back(weights);
	}
}

void ProgMovieEstimateGain::show()
{
	if (verbose==0)
		return;
	std::cout
	<< "Input movie:     " << fnIn            << std::endl
	<< "Output rootname: " << fnRoot          << std::endl
	<< "N. Iterations:   " << Niter           << std::endl
	<< "Sigma:           " << sigma           << std::endl
	<< "Max. Sigma:      " << maxSigma        << std::endl
	<< "Sigma step:      " << sigmaStep       << std::endl
	<< "Single ref:      " << singleReference << std::endl
	<< "Frame step:      " << frameStep       << std::endl
	<< "Ext. gain image: " << fnGain          << std::endl
	<< "Apply ext. gain: " << applyGain       << std::endl
	;
}


void ProgMovieEstimateGain::run()
{
	produceSideInfo();

	FileName fnFrame;
	Image<double> Iframe;
	MultidimArray<double> IframeTransformed, IframeIdeal;
	MultidimArray<double> sumIdeal;
	MultidimArray<double> &mICorrection=ICorrection();

	if (applyGain)
	{
            int idx=1;
            FOR_ALL_OBJECTS_IN_METADATA(mdIn)
            {
                mdIn.getValue(MDL_IMAGE,fnFrame,__iter.objId);
                Iframe.read(fnFrame);
                MultidimArray<double> &mIframe = Iframe();
                FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mIframe)
                    DIRECT_MULTIDIM_ELEM(mIframe,n) /= DIRECT_MULTIDIM_ELEM(mICorrection,n);
                //Iframe.write(formatString("%i@%s"%(idx,fnCorrected)));
                Iframe.write(fnCorrected, idx, WRITE_REPLACE);
            }
	}else{
        for (int n=0; n<Niter; n++)
        {
            std::cout << "Iteration " << n << std::endl;
            sumIdeal.initZeros(Ydim,Xdim);
            int iFrame=0;
            FOR_ALL_OBJECTS_IN_METADATA(mdIn)
            {
                if (iFrame%frameStep==0)
                {
                    mdIn.getValue(MDL_IMAGE,fnFrame,__iter.objId);
                    std::cout << "   Frame " << fnFrame << std::endl;
                    Iframe.read(fnFrame);
                    IframeIdeal = Iframe();
                    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(IframeIdeal)
                        DIRECT_A2D_ELEM(IframeIdeal,i,j)/=DIRECT_A2D_ELEM(mICorrection,i,j);
                    computeHistograms(IframeIdeal);

                    //sigma=0;
                    size_t bestSigmaCol = 0;
                    if (sigma>=0)
                    {
                    	double bestDistance=1e38;
                    	for (size_t s = 0; s< listOfWeights.size(); ++s)
                    		if (fabs(listOfSigmas[s]-sigma)<bestDistance)
                    		{
                    			bestDistance=fabs(listOfSigmas[s]-sigma);
                    			bestSigmaCol=s;
                    		}
                    }
                    else
                    	selectBestSigmaByColumn(IframeIdeal);
                    std::cout << "      sigmaCol: " << listOfSigmas[bestSigmaCol] << std::endl;
                    size_t bestSigmaRow = 0;

                    if (sigma>=0)
                    {
                    	double bestDistance=1e38;
                    	for (size_t s = 0; s< listOfWeights.size(); ++s)
                    		if (fabs(listOfSigmas[s]-sigma)<bestDistance)
                    		{
                    			bestDistance=fabs(listOfSigmas[s]-sigma);
                    			bestSigmaRow=s;
                    		}
                    }
                    else
                    	selectBestSigmaByRow(IframeIdeal);
                    std::cout << "      sigmaRow: " << listOfSigmas[bestSigmaRow] << std::endl;

                    constructSmoothHistogramsByRow(listOfWeights[bestSigmaRow],listOfWidths[bestSigmaRow]);
                    transformGrayValuesRow(IframeIdeal,IframeTransformed);
                    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(IframeTransformed)
                        DIRECT_A2D_ELEM(sumIdeal,i,j)+=DIRECT_A2D_ELEM(IframeTransformed,i,j);
                    constructSmoothHistogramsByColumn(listOfWeights[bestSigmaCol],listOfWidths[bestSigmaCol]);
                    transformGrayValuesColumn(IframeIdeal,IframeTransformed);
                    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(IframeTransformed)
                        DIRECT_A2D_ELEM(sumIdeal,i,j)+=DIRECT_A2D_ELEM(IframeTransformed,i,j);
                 }
                 iFrame++;
            }

            FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(mICorrection)
            {
                double den=DIRECT_A2D_ELEM(sumObs,i,j);
                if (fabs(den)<1e-6)
                    DIRECT_A2D_ELEM(mICorrection,i,j)=1.0;
                else
                    DIRECT_A2D_ELEM(mICorrection,i,j)=DIRECT_A2D_ELEM(sumIdeal,i,j)/den;
            }
            mICorrection/=mICorrection.computeAvg();

    #ifdef NEVER_DEFINED
        ICorrection.write(fnCorr);
        Image<double> save;
        typeCast(sumIdeal,save());
        save.write("PPPSumIdeal.xmp");
        typeCast(sumObs,save());
        save.write("PPPSumObs.xmp");
        //std::cout << "Press any key\n";
        //char c; std::cin >> c;
    #endif
        }
    }
	ICorrection.write(fnRoot+"_correction.xmp");
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(mICorrection)
		if (DIRECT_A2D_ELEM(mICorrection,i,j)>1e-5)
			DIRECT_A2D_ELEM(mICorrection,i,j)=1.0/DIRECT_A2D_ELEM(mICorrection,i,j);
		else
			DIRECT_A2D_ELEM(mICorrection,i,j)=1;
	mICorrection/=mICorrection.computeAvg();
	ICorrection.write(fnRoot+"_gain.xmp");
}

void ProgMovieEstimateGain::computeHistograms(const MultidimArray<double> &Iframe)
{
		double* auxElemC=new double[Ydim];
		double* auxElemR=new double[Xdim];

		for(size_t j=0; j<XSIZE(columnH); j++)
		{
			for(size_t i=0; i<Ydim; i++)
				auxElemC[i]=A2D_ELEM(Iframe,i,j);
			std::sort(auxElemC, auxElemC+Ydim);
			for(size_t i=0; i<Ydim; i++)
				A2D_ELEM(columnH,i,j)=auxElemC[i];
		}
		delete[] auxElemC;

		for(size_t i=0; i<YSIZE(rowH); i++)
		{
			memcpy(auxElemR,&A2D_ELEM(Iframe,i,0),Xdim*sizeof(double));
			std::sort(auxElemR, auxElemR+Xdim);
			memcpy(&A2D_ELEM(rowH,i,0),auxElemR,Xdim*sizeof(double));
		}
		delete[] auxElemR;


#ifdef NEVER_DEFINED
	Image<double> save;
	typeCast(columnH,save());
	save.write("PPPcolumnH.xmp");
	typeCast(rowH,save());
	save.write("PPProwH.xmp");
#endif
}

void ProgMovieEstimateGain::constructSmoothHistogramsByColumn(const double *listOfWeights, int width)
{

	smoothColumnH.initZeros(columnH);

	for (size_t j=0; j<XSIZE(columnH); ++j)
	{
		double sumWeightsC = 0;
		for(int k = -width; k<=width; ++k)
		{
			if (j+k<0 || j+k>=XSIZE(columnH))
				continue;
			double actualWeightC = listOfWeights[abs(k)];
			sumWeightsC += actualWeightC;
			for (size_t i=0; i<Ydim; ++i)
				DIRECT_A2D_ELEM(smoothColumnH,i,j) += actualWeightC * DIRECT_A2D_ELEM(columnH,i,j+k);
		}

		double iSumWeightsC=1/sumWeightsC;
		for (size_t i=0; i<Ydim; ++i)
			DIRECT_A2D_ELEM(smoothColumnH,i,j) *= iSumWeightsC;
	}

	if (singleReference)
	{
		// Compute the average of all column histograms
		for (size_t i=0; i<Ydim; ++i)
			for (size_t j=1; j<Xdim; ++j)
				DIRECT_A2D_ELEM(smoothColumnH,i,0)+=DIRECT_A2D_ELEM(smoothColumnH,i,j);

		double iXdim=1.0/Xdim;
		for (size_t i=0; i<Ydim; ++i)
		{
			DIRECT_A2D_ELEM(smoothColumnH,i,0)*=iXdim;
			double aux=DIRECT_A2D_ELEM(smoothColumnH,i,0);
			for (size_t j=1; j<Xdim; ++j)
				DIRECT_A2D_ELEM(smoothColumnH,i,j)=aux;
		}
	}
#ifdef NEVER_DEFINED
	Image<double> save;
	typeCast(smoothColumnH,save());
	save.write("PPPsmoothColumnH.xmp");
#endif
}

void ProgMovieEstimateGain::constructSmoothHistogramsByRow(const double *listOfWeights, int width)
{
	smoothRowH.initZeros(rowH);
	for(size_t i = 0; i<YSIZE(rowH); ++i)
	{
		double sumWeightsR = 0;
		for(int k = -width; k<=width; ++k)
		{
			if (i+k<0 || i+k>=YSIZE(rowH))
				continue;
			double actualWeightR = listOfWeights[abs(k)];
			sumWeightsR += actualWeightR;
			for (size_t j=0; j< Xdim; ++j)
				DIRECT_A2D_ELEM(smoothRowH,i,j) += actualWeightR * DIRECT_A2D_ELEM(rowH,i+k,j);
		}
		double iSumWeightsR=1/sumWeightsR;
		for (size_t j=0; j<Xdim; ++j)
			DIRECT_A2D_ELEM(smoothRowH,i,j) *= iSumWeightsR;
	}

	if (singleReference)
	{
		// Compute the average of all row histograms
		for (size_t j=0; j<Xdim; ++j)
			for (size_t i=1; i<Ydim; ++i)
				DIRECT_A2D_ELEM(smoothRowH,0,j)+=DIRECT_A2D_ELEM(smoothRowH,i,j);

		double iYdim=1.0/Ydim;
		for (size_t j=0; j<Xdim; ++j)
		{
			DIRECT_A2D_ELEM(smoothRowH,0,j)*=iYdim;
			double aux=DIRECT_A2D_ELEM(smoothRowH,0,j);
			for (size_t i=1; i<Ydim; ++i)
				DIRECT_A2D_ELEM(smoothRowH,i,j)=aux;
		}
	}

#ifdef NEVER_DEFINED
	Image<double> save;
	typeCast(smoothRowH,save());
	save.write("PPPsmoothRowH.xmp");
#endif
}
#undef NEVER_DEFINED

void ProgMovieEstimateGain::transformGrayValuesColumn(const MultidimArray<double> &Iframe, MultidimArray<double> &IframeTransformedColumn)
{
	IframeTransformedColumn.initZeros(Ydim,Xdim);
	aSingleColumnH.resizeNoCopy(Ydim);
	double *aSingleColumnH0=&DIRECT_A1D_ELEM(aSingleColumnH,0);
	double *aSingleColumnHF=(&DIRECT_A1D_ELEM(aSingleColumnH,Ydim-1))+1;


	for (size_t j=0; j<Xdim; ++j)
	{
		for (size_t i=0; i<Ydim; ++i)
			DIRECT_A1D_ELEM(aSingleColumnH,i)=DIRECT_A2D_ELEM(columnH,i,j);

		for (size_t i=0; i<Ydim; ++i)
		{
			double pixval=DIRECT_A2D_ELEM(Iframe,i,j);
			double *pixvalPtr=std::upper_bound(aSingleColumnH0,aSingleColumnHF,pixval);
			pixvalPtr-=1;
			int idx=pixvalPtr-aSingleColumnH0;
			DIRECT_A2D_ELEM(IframeTransformedColumn,i,j)+=(double)DIRECT_A2D_ELEM(smoothColumnH,idx,j);
		}
	}


#ifdef NEVER_DEFINED
	Image<double> save;
	typeCast(IframeTransformedColumn,save());
	save.write("PPPIframeTransformedColumn.xmp");
	typeCast(Iframe,save());
	save.write("PPPIframe.xmp");
	std::cout << "Press any key to continue\n";
	char c; std::cin >> c;
#endif
}


void ProgMovieEstimateGain::transformGrayValuesRow(const MultidimArray<double> &Iframe, MultidimArray<double> &IframeTransformedRow)
{
	IframeTransformedRow.initZeros(Ydim,Xdim);
	aSingleRowH.resizeNoCopy(Xdim);
	double *aSingleRowH0=&DIRECT_A1D_ELEM(aSingleRowH,0);
	double *aSingleRowHF=(&DIRECT_A1D_ELEM(aSingleRowH,Xdim-1))+1;

	for (size_t i=0; i<Ydim; ++i)
	{
		memcpy(aSingleRowH0,&DIRECT_A2D_ELEM(rowH,i,0),Xdim*sizeof(double));

			for (size_t j=0; j<Xdim; ++j)
			{
				double pixvalR=DIRECT_A2D_ELEM(Iframe,i,j);
				double *pixvalPtrR=std::upper_bound(aSingleRowH0,aSingleRowHF,pixvalR);
				pixvalPtrR-=1;
				int idxR=pixvalPtrR-aSingleRowH0;
				DIRECT_A2D_ELEM(IframeTransformedRow,i,j)+=(double)DIRECT_A2D_ELEM(smoothRowH,i,idxR);
			}
	}

#ifdef NEVER_DEFINED
	Image<double> save;
	typeCast(IframeTransformedRow,save());
	save.write("PPPIframeTransformedRow.xmp");
	typeCast(Iframe,save());
	save.write("PPPIframe.xmp");
	std::cout << "Press any key to continue\n";
	char c; std::cin >> c;
#endif

}

double computeTVColumns(MultidimArray<double> &I)
{
	double retvalC=0;
	for (size_t i=0; i<YSIZE(I); ++i)
	    for (size_t j=0; j<XSIZE(I)-1; ++j)
	    	retvalC+=std::abs(DIRECT_A2D_ELEM(I,i,j)-DIRECT_A2D_ELEM(I,i,j+1));

	return retvalC/((XSIZE(I)-1)*YSIZE(I));
}

double computeTVRows(MultidimArray<double> &I)
{
	double retvalR=0;
	for (size_t i=0; i<YSIZE(I)-1; ++i)
	    for (size_t j=0; j<XSIZE(I); ++j)
	    	retvalR+=std::abs(DIRECT_A2D_ELEM(I,i,j)-DIRECT_A2D_ELEM(I,i+1,j));

	return retvalR/((YSIZE(I)-1)*XSIZE(I));
}

size_t ProgMovieEstimateGain::selectBestSigmaByColumn(const MultidimArray<double> &Iframe)
{
	double bestAvgTV=1e38;
	size_t best_s=0;
	MultidimArray<double> IframeTransformed;

	for(size_t s = 0; s< listOfWeights.size(); ++s)
	{
		constructSmoothHistogramsByColumn(listOfWeights[s],listOfWidths[s]);
		transformGrayValuesColumn(Iframe,IframeTransformed);
		double avgTV=computeTVColumns(IframeTransformed);
		if (avgTV<bestAvgTV)
		{
			bestAvgTV=avgTV;
			best_s=s;
		}
	}
	return best_s;
}

size_t ProgMovieEstimateGain::selectBestSigmaByRow(const MultidimArray<double> &Iframe)
{
	double bestAvgTV=1e38;
	size_t best_s=0;
	MultidimArray<double> IframeTransformed;

	for(size_t s = 0; s< listOfWeights.size(); ++s)
	{
		constructSmoothHistogramsByRow(listOfWeights[s],listOfWidths[s]);
		transformGrayValuesRow(Iframe,IframeTransformed);
		double avgTV=computeTVRows(IframeTransformed);
		if (avgTV<bestAvgTV)
		{
			bestAvgTV=avgTV;
			best_s=s;
		}
	}

	return best_s;
}
