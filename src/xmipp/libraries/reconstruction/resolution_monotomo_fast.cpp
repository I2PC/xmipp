/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 * 			   Carlos Oscar S. Sorzano            coss@cnb.csic.es (2019)
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

#include "resolution_monotomo_fast.h"
#include <core/bilib/kernel.h>
#include "core/linear_system_helper.h"
//#define DEBUG
//#define DEBUG_MASK
//#define TEST_FRINGES



void ProgMonoTomoFast::readParams()
{
	fnVol = getParam("--vol");
	fnVol2 = getParam("--vol2");
	fnMeanVol = getParam("--meanVol");
	fnOut = getParam("-o");
	fnFilt = getParam("--filteredMap");
	fnMask = getParam("--mask");
	sampling = getDoubleParam("--sampling_rate");
	fnmaskWedge = getParam("--maskWedge");
	minRes = getDoubleParam("--minRes");
	maxRes = getDoubleParam("--maxRes");
	freq_step = getDoubleParam("--step");
	trimBound = getDoubleParam("--trimmed");
	significance = getDoubleParam("--significance");
	nthrs = getIntParam("--threads");
}


void ProgMonoTomoFast::defineParams()
{
	addUsageLine("This function determines the local resolution of a tomogram. It makes use of two reconstructions, odd and even. The difference between them"
			"gives a noise reconstruction. Thus, by computing the local amplitude of the signal at different frequencies and establishing a comparison with"
			"the noise, the local resolution is computed");
	addParamsLine("  --vol <vol_file=\"\">   			: Half volume 1");
	addParamsLine("  --vol2 <vol_file=\"\">				: Half volume 2");
	addParamsLine("  [--mask <vol_file=\"\">]  			: Mask defining the signal. ");
	addParamsLine("  -o <output=\"MGresolution.vol\">	: Local resolution volume (in Angstroms)");
	addParamsLine("  [--maskWedge <vol_file=\"\">]	: Mask containing the missing wedge in Fourier space");
	addParamsLine("  [--filteredMap <vol_file=\"\">]	            : Local resolution volume filtered considering the missing wedge (in Angstroms)");
	addParamsLine("  --meanVol <vol_file=\"\">			: Mean volume of half1 and half2 (only it is necessary the two haves are used)");
	addParamsLine("  [--sampling_rate <s=1>]   			: Sampling rate (A/px)");
	addParamsLine("  [--step <s=0.25>]       			: The resolution is computed at a number of frequencies between minimum and");
	addParamsLine("                            			: maximum resolution px/A. This parameter determines that number");
	addParamsLine("  [--minRes <s=30>]         			: Minimum resolution (A)");
	addParamsLine("  [--maxRes <s=1>]         			: Maximum resolution (A)");
	addParamsLine("  [--trimmed <s=0.5>]       			: Trimming percentile");
	addParamsLine("  [--significance <s=0.95>]       	: The level of confidence for the hypothesis test.");
	addParamsLine("  [--threads <s=4>]               	: Number of threads");
}


void ProgMonoTomoFast::readAndPrepareData()
{
	std::cout << "Starting..." << std::endl;
	std::cout << "           " << std::endl;
	std::cout << "IMPORTANT: If the angular step of the tilt series is higher than 3 degrees"<< std::endl;
	std::cout << "           then, the tomogram is not properly for MonoTomo. Despite this is not "<< std::endl;
	std::cout << "           optimal, MonoTomo will try to compute the local resolution." << std::endl;
	std::cout << "           " << std::endl;

	Image<float> V;
	auto &inputVol = V();

	Image<float> V1, V2;
	V1.read(fnVol);
	V2.read(fnVol2);
	V()=0.5*(V1()+V2());
	//V.write(fnMeanVol);

	V().setXmippOrigin();

	size_t Ndim;
	inputVol.getDimensions(Xdim, Ydim, Zdim, Ndim);

	auto hw = new CPU((unsigned) nthrs); //number of threads
    hw->set();
	auto patchDim = Dimensions(XSIZE(inputVol), YSIZE(inputVol), ZSIZE(inputVol));
	auto settings = FFTSettingsNew<float>(patchDim);
    plan = transformer_inv.createPlan(*hw, settings, true); // true means that it also be used by the inputdata
	fftV = (std::complex<float>*) transformer_inv.allocateAligned(settings.fBytesBatch());
	transformer_inv.init(*hw, settings);
	transformer_inv.fft(inputVol.data, fftV);



	//VRiesz.resizeNoCopy(inputVol);
	



	auto transformer  = FFTwT<float>();

	#ifdef TEST_FRINGES

	double modulus, xx, yy, zz;

	long nnn=0;
	for(size_t k=0; k<ZSIZE(inputVol); ++k)
	{
		zz = (k-ZSIZE(inputVol)*0.5)*(k-ZSIZE(inputVol)*0.5);
		for(size_t i=0; i<YSIZE(inputVol); ++i)
		{
			yy = (i-YSIZE(inputVol)*0.5)*(i-YSIZE(inputVol)*0.5);
			for(size_t j=0; j<XSIZE(inputVol); ++j)
			{
				xx = (j-XSIZE(inputVol)*0.5)*(j-XSIZE(inputVol)*0.5);
				DIRECT_MULTIDIM_ELEM(inputVol,nnn) = cos(0.1*sqrt(xx+yy+zz));
				++nnn;
			}
		}
	}

	Image<double> saveiu;
	saveiu = inputVol;
	saveiu.write("franjas.vol");
	exit(0);
	#endif



    //auto plan = transformer_inv.createPlan(*hw, settings, true); // true means that it also be used by the inputdata
	// fftV = (std::complex<float>*) transformer.allocateAligned(settings.fBytesBatch());
	// transformer.init(*hw, settings);
	// transformer.fft(inputVol.data, fftV);

	// TODO: See size of fftV for odd dimensions
	ZdimFT = ZSIZE(inputVol);
	YdimFT = YSIZE(inputVol);
	XdimFT = XSIZE(inputVol)/2 + 1;



	#ifdef DEBUG
	Image<double> saveiu;
	saveiu = 1/iu;
	saveiu.write("iu.vol");
	#endif



	// Prepare mask
	MultidimArray<int> &pMask=mask();

	if (fnMask != "")
	{
		mask.read(fnMask);
		mask().setXmippOrigin();
	}
	else
	{
		pMask.resizeNoCopy(1, Zdim, Ydim, Xdim);
		pMask.initConstant(1);
	}

	NVoxelsOriginalMask = 0;

	FOR_ALL_ELEMENTS_IN_ARRAY3D(pMask)
	{
		if (A3D_ELEM(pMask, k, i, j) == 1)
			++NVoxelsOriginalMask;
//		if (i*i+j*j+k*k > R*R)
//			A3D_ELEM(pMask, k, i, j) = -1;
	}


	V1.read(fnVol);
	V2.read(fnVol2);

	V1()-=V2();
	V1()/=2;


	#ifdef DEBUG
	  V1.write("diff_volume.vol");
	#endif

	int N_smoothing = 10;

	int siz_z = Zdim*0.5;
	int siz_y = Ydim*0.5;
	int siz_x = Xdim*0.5;


	int limit_distance_x = (siz_x-N_smoothing);
	int limit_distance_y = (siz_y-N_smoothing);
	int limit_distance_z = (siz_z-N_smoothing);

	long n=0;
	float uz, uy, ux;
	for(int k=0; k<Zdim; ++k)
	{
		uz = (k - siz_z);
		for(int i=0; i<Ydim; ++i)
		{
			uy = (i - siz_y);
			for(int j=0; j<Xdim; ++j)
			{
				ux = (j - siz_x);

				if (abs(ux)>=limit_distance_x)
				{
					DIRECT_MULTIDIM_ELEM(V1(), n) *= 0.5*(1+cos(PI*(limit_distance_x - abs(ux))/N_smoothing));
					DIRECT_MULTIDIM_ELEM(pMask, n) = 0;
				}
				if (abs(uy)>=limit_distance_y)
				{
					DIRECT_MULTIDIM_ELEM(V1(), n) *= 0.5*(1+cos(PI*(limit_distance_y - abs(uy))/N_smoothing));
					DIRECT_MULTIDIM_ELEM(pMask, n) = 0;
				}
				if (abs(uz)>=limit_distance_z)
				{
					DIRECT_MULTIDIM_ELEM(V1(), n) *= 0.5*(1+cos(PI*(limit_distance_z - abs(uz))/N_smoothing));
					DIRECT_MULTIDIM_ELEM(pMask, n) = 0;
				}
				++n;
			}
		}
	}

	

	auto transformer2  = FFTwT<float>();
	fftN = (std::complex<float>*) transformer2.allocateAligned(settings.fBytesBatch());
	transformer2.init(*hw, settings);
	transformer2.fft(V1().data, fftN);

	V.clear();

	frequencyMap();

// 	Image<float> img;
// 	img() = iu;
// 	img.write("iu.mrc");
}

void ProgMonoTomoFast::frequencyMap()
{
	double u;

	freq_fourier_z.initZeros(ZdimFT);
	freq_fourier_x.initZeros(XdimFT);
	freq_fourier_y.initZeros(YdimFT);

	VEC_ELEM(freq_fourier_z,0) = 1e-38;
	for(size_t k=0; k<ZdimFT; ++k)
	{
		FFT_IDX2DIGFREQ(k,Zdim, u);
		VEC_ELEM(freq_fourier_z,k) = (float) u;
	}

	VEC_ELEM(freq_fourier_y,0) = 1e-38;
	for(size_t k=0; k<YdimFT; ++k)
	{
		FFT_IDX2DIGFREQ(k,Ydim, u);
		VEC_ELEM(freq_fourier_y,k) = (float) u;
	}

	VEC_ELEM(freq_fourier_x,0) = 1e-38;
	for(size_t k=0; k<XdimFT; ++k)
	{
		FFT_IDX2DIGFREQ(k,Xdim, u);
		VEC_ELEM(freq_fourier_x,k) = (float) u;
	}

	iu.initZeros(1, Zdim, Ydim, Xdim);
	// Calculate u and first component of Riesz vector
	float uz, uy, ux, uz2, u2, uz2y2;
	long n=0;
	for(size_t k=0; k<ZdimFT; ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz2=uz*uz;

		for(size_t i=0; i<YdimFT; ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uz2y2=uz2+uy*uy;

			for(size_t j=0; j<XdimFT; ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				u2=uz2y2+ux*ux;
				if ((k != 0) || (i != 0) || (j != 0))
					DIRECT_MULTIDIM_ELEM(iu,n) = 1.0/sqrt(u2);
				else
					DIRECT_MULTIDIM_ELEM(iu,n) = 1e38;
				++n;
			}
		}
	}
}

void ProgMonoTomoFast::amplitudeMonogenicSignal3D(float freq, float freqH, float freqL, float *amplitude, int count, FileName fnDebug)
{
 	std::complex<float> J(0,1);

 	// Filter the input volume and add it to amplitude
 	float ideltal=PI/(freq-freqH);

	long FTelems = XdimFT*YdimFT*ZdimFT;
	long Realelems = Xdim*Ydim*Zdim;

	std::complex<float>* fftVRiesz = new std::complex<float>[FTelems]();
	std::complex<float>* fftVRiesz_aux = new std::complex<float>[FTelems]();

	for (long n=0; n<FTelems; n++)
	{
		auto iun = iu[n];

		float un=1.0/iun;

		if (freqH<=un && un<=freq)
		{
			float H;
			std::complex<float> aux;
			H = 0.5*(1+cos((un-freq)*ideltal));
			aux = fftV[n]*H;
			
			fftVRiesz[n] = aux;
			fftVRiesz_aux[n] = -J*aux*iun;
		} 
		else if (un>freq)
		{
			std::complex<float> aux2 = -J*fftV[n];
			fftVRiesz[n] = aux2;
			fftVRiesz_aux[n] = -J*aux2*iun;
		}
	}

	Image<float> realMap(XdimFT, YdimFT, ZdimFT, 1);
	for (long n=0; n<FTelems; n++)
	{
		float realValue = real(fftV[n]);
		realMap().data[n] = realValue;
	}
	realMap.write("relMap.mrc");


	transformer_inv.ifft(plan, fftVRiesz, VRiesz);

	
	MultidimArray<float> tt(1, Zdim, Ydim, Xdim, VRiesz);
	Image<float> ttimg(tt);
	ttimg.write("tt.mrc");

	#ifdef DEBUG
	Image<double> filteredvolume;
	filteredvolume = VRiesz;
	filteredvolume.write(formatString("Volumen_filtrado_%i.vol", count));

	FileName iternumber;
	iternumber = formatString("_Volume_%i.vol", count);
	Image<double> saveImg2;
	saveImg2() = VRiesz;
	  if (fnDebug.c_str() != "")
	  {
		saveImg2.write(fnDebug+iternumber);
	  }
	saveImg2.clear(); 
	#endif


	for (long n=0; n<Realelems; n++)
	{
		float auxV = VRiesz[n];
		amplitude[n] = auxV*auxV;
	}

	// Calculate first component of Riesz vector
	float uz, uy, ux;
	long n=0;
	for(size_t k=0; k<ZdimFT; ++k)
	{
		for(size_t i=0; i<YdimFT; ++i)
		{
			for(size_t j=0; j<XdimFT; ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				fftVRiesz[n] = ux*fftVRiesz_aux[n];
				++n;
			}
		}
	}

	transformer_inv.ifft(plan, fftVRiesz, VRiesz);

	for (long n=0; n<Realelems; n++)
	{
		float auxV = VRiesz[n];
		amplitude[n] += auxV*auxV;
	}

	// Calculate second and third components of Riesz vector
	n=0;
	for(size_t k=0; k<ZdimFT; ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		for(size_t i=0; i<YdimFT; ++i)
		{
			uy = VEC_ELEM(freq_fourier_y,i);
			for(size_t j=0; j<XdimFT; ++j)
			{
				fftVRiesz[n] = uy*fftVRiesz_aux[n];
				fftVRiesz_aux[n] = uz*fftVRiesz_aux[n];
				++n;
			}
		}
	}

	transformer_inv.ifft(plan, fftVRiesz, VRiesz);

		for (long n=0; n<Realelems; n++)
	{
		float auxV = VRiesz[n];
		amplitude[n] += auxV*auxV;
	}

	transformer_inv.ifft(plan, fftVRiesz_aux, VRiesz);

		for (long n=0; n<Realelems; n++)
	{
		float auxV = VRiesz[n];
		amplitude[n] += auxV*auxV;
	}
/*
// 	#ifdef DEBUG
// 	if (fnDebug.c_str() != "")
// 	{
// 	Image<double> saveImg;
// 	saveImg = amplitude;
// 	FileName iternumber;
// 	iternumber = formatString("_Amplitude_%i.vol", count);
// 	saveImg.write(fnDebug+iternumber);
// 	saveImg.clear();
// 	}
// 	#endif // DEBUG
// //
// 	// Low pass filter the monogenic amplitude
// 	transformer_inv.FourierTransform(amplitude, fftVRiesz, false);
// 	double raised_w = PI/(freqL-freq);

// 	n=0;

// 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftVRiesz)
// 	{
// 		double un=1.0/DIRECT_MULTIDIM_ELEM(iu,n);
// 		if (freqL>=un && un>=freq)
// 		{
// 			DIRECT_MULTIDIM_ELEM(fftVRiesz,n) *= 0.5*(1 + cos(raised_w*(un-freq)));
// 		}
// 		else
// 		{
// 			if (un>freqL)
// 			{
// 				DIRECT_MULTIDIM_ELEM(fftVRiesz,n) = 0;
// 			}
// 		}
// 	}
// 	transformer_inv.inverseFourierTransform();


// 	#ifdef DEBUG
// 	Image<double> saveImg2;
// 	saveImg2 = amplitude;
// 	FileName fnSaveImg2;
// 	if (fnDebug.c_str() != "")
// 	{
// 		iternumber = formatString("_Filtered_Amplitude_%i.vol", count);
// 		saveImg2.write(fnDebug+iternumber);
// 	}
// 	saveImg2.clear();
// 	#endif // DEBUG
*/
	delete []fftVRiesz;
	delete []fftVRiesz_aux;
}

void ProgMonoTomoFast::resolution2eval(int &count_res, double step,
								double &resolution, double &last_resolution,
								double &freq, double &freqL,
								int &last_fourier_idx,
								bool &continueIter,	bool &breakIter)
{
	resolution = maxRes - count_res*step;
	freq = sampling/resolution;
	++count_res;

	double Nyquist = 2*sampling;
	double aux_frequency;
	int fourier_idx;

	DIGFREQ2FFT_IDX(freq, Zdim, fourier_idx);

	FFT_IDX2DIGFREQ(fourier_idx, Zdim, aux_frequency);

	freq = aux_frequency;

	if (fourier_idx == last_fourier_idx)
	{
		continueIter = true;
		return;
	}

	last_fourier_idx = fourier_idx;
	resolution = sampling/aux_frequency;


	if (count_res == 0)
		last_resolution = resolution;

	if (resolution<Nyquist)
	{
		breakIter = true;
		return;
	}

	freqL = sampling/(resolution + step);

	int fourier_idx_2;

	DIGFREQ2FFT_IDX(freqL, Zdim, fourier_idx_2);

	if (fourier_idx_2 == fourier_idx)
	{
		if (fourier_idx > 0){
			FFT_IDX2DIGFREQ(fourier_idx - 1, Zdim, freqL);
		}
		else{
			freqL = sampling/(resolution + step);
		}
	}

}

void ProgMonoTomoFast::localNoise(float *noiseMap, Matrix2D<double> &noiseMatrix, int boxsize, Matrix2D<float> &thresholdMatrix)
{
//	std::cout << "Analyzing local noise" << std::endl;

	int Nx = Xdim/boxsize;
	int Ny = Ydim/boxsize;

	noiseMatrix.initZeros(Ny, Nx);

	// For the spline regression
	int lX=std::min(8,Nx-2), lY=std::min(8,Ny-2);
    WeightedLeastSquaresHelper helper;
    helper.A.initZeros(Nx*Ny,lX*lY);
    helper.b.initZeros(Nx*Ny);
    helper.w.initZeros(Nx*Ny);
    helper.w.initConstant(1);
    double hX = Xdim / (double)(lX-3);
    double hY = Ydim / (double)(lY-3);

	if ( (Xdim<boxsize) || (Ydim<boxsize) )
		std::cout << "Error: The tomogram in x-direction or y-direction is too small" << std::endl;

	std::vector<float> noiseVector(1);
	std::vector<double> x,y,t;

	int xLimit, yLimit, xStart, yStart;
	int startX, startY, startZ, finishX, finishY, finishZ;
	startX = 0;
	startY = 0;
	startZ = 0;
	finishX = Xdim;
	finishY = Ydim;
	finishZ = Zdim;

	long counter;
    int idxBox=0;

	for (int X_boxIdx=0; X_boxIdx<Nx; ++X_boxIdx)
	{
		if (X_boxIdx==Nx-1)
		{
			xStart = startX + X_boxIdx*boxsize;
			xLimit = finishX;
		}
		else
		{
			xStart = startX + X_boxIdx*boxsize;
			xLimit = startX + (X_boxIdx+1)*boxsize;
		}

		for (int Y_boxIdx=0; Y_boxIdx<Ny; ++Y_boxIdx)
		{
			if (Y_boxIdx==Ny-1)
			{
				yStart = startY + Y_boxIdx*boxsize;
				yLimit =  finishY;
			}
			else
			{
				yStart = startY + Y_boxIdx*boxsize;
				yLimit = startY + (Y_boxIdx+1)*boxsize;
			}

			counter = 0;
			long n = 0;
			for (int k = startZ; k<finishZ; k++)
			{
				for (int i = yStart; i<yLimit; i++)
				{
					for (int j = xStart; j<xLimit; j++)
					{
						if (n%257 == 0)
						{
							noiseVector.push_back( noiseMap[n]);
						} //we take one voxel each 257 (prime number) points to reduce noise data
//							noiseVector.push_back( A3D_ELEM(noiseMap, k, i, j) );
						n++;
					}
				}
			}

			std::sort(noiseVector.begin(),noiseVector.end());
			MAT_ELEM(noiseMatrix, Y_boxIdx, X_boxIdx) = (double ) noiseVector[size_t(noiseVector.size()*significance)];

			double tileCenterY=0.5*(yLimit+yStart)-startY; // Translated to physical coordinates
			double tileCenterX=0.5*(xLimit+xStart)-startX;
			// Construction of the spline equation system
			long idxSpline=0;
			for(int controlIdxY = -1; controlIdxY < (lY - 1); ++controlIdxY)
			{
				double tmpY = Bspline03((tileCenterY / hY) - controlIdxY);
				VEC_ELEM(helper.b,idxBox)=MAT_ELEM(noiseMatrix, Y_boxIdx, X_boxIdx);
				if (tmpY == 0.0)
				{
					idxSpline+=lX;
					continue;
				}

				for(int controlIdxX = -1; controlIdxX < (lX - 1); ++controlIdxX)
				{
					double tmpX = Bspline03((tileCenterX / hX) - controlIdxX);
					MAT_ELEM(helper.A,idxBox,idxSpline) = tmpY * tmpX;
					idxSpline+=1;
				}
			}
			x.push_back(tileCenterX);
			y.push_back(tileCenterY);
			t.push_back(MAT_ELEM(noiseMatrix, Y_boxIdx, X_boxIdx));
			noiseVector.clear();
			idxBox+=1;
		}
	}


	// Spline coefficients
	Matrix1D<double> cij;
	weightedLeastSquares(helper, cij);

	thresholdMatrix.initZeros(Ydim, Xdim);

	for (int i=0; i<Ydim; ++i)
	{
		for (int j=0; j<Xdim; ++j)
		{
			long idxSpline=0;

			for(int controlIdxY = -1; controlIdxY < (lY - 1); ++controlIdxY)
			{
				double tmpY = Bspline03((i / hY) - controlIdxY);

				if (tmpY == 0.0)
				{
					idxSpline+=lX;
					continue;
				}

				double xContrib=0.0;
				for(int controlIdxX = -1; controlIdxX < (lX - 1); ++controlIdxX)
				{
					double tmpX = Bspline03((j / hX) - controlIdxX);
					xContrib+=VEC_ELEM(cij,idxSpline) * tmpX;// *tmpY;
					idxSpline+=1;
				}
				MAT_ELEM(thresholdMatrix,i,j)+=(float) xContrib*tmpY;
			}
		}
	}
}


void ProgMonoTomoFast::lowestResolutionbyPercentile(MultidimArray<double> &resolutionVol,
		std::vector<float> &list, double &cut_value, double &resolutionThreshold)
{
	double last_resolution_2 = list[(list.size()-1)];

	double lowest_res;
	lowest_res = list[0];

	// Count number of voxels with resolution
	size_t N=0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
		if (DIRECT_MULTIDIM_ELEM(resolutionVol, n)>=(last_resolution_2-0.001))//&& (DIRECT_MULTIDIM_ELEM(resolutionVol, n)<lowest_res) ) //the value 0.001 is a tolerance
			++N;

	// Get all resolution values
	MultidimArray<double> resolutions(N);
	size_t N_iter=0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
		if (DIRECT_MULTIDIM_ELEM(resolutionVol, n)>(last_resolution_2-0.001))//&& (DIRECT_MULTIDIM_ELEM(resolutionVol, n)<lowest_res))
		{
			DIRECT_MULTIDIM_ELEM(resolutions,N_iter++)=DIRECT_MULTIDIM_ELEM(resolutionVol, n);
		}

//	median = resolutionVector[size_t(resolutionVector.size()*0.5)];

	// Sort value and get threshold
	std::sort(&A1D_ELEM(resolutions,0),&A1D_ELEM(resolutions,N));
	double filling_value = A1D_ELEM(resolutions, (int)(0.5*N)); //median value
	double trimming_value = A1D_ELEM(resolutions, (int)((1-cut_value)*N));
	resolutionThreshold = A1D_ELEM(resolutions, (int)((0.95)*N));

	std::cout << "resolutionThreshold = " << resolutionThreshold <<  std::endl;
}

void ProgMonoTomoFast::postProcessingLocalResolutions(MultidimArray<double> &resolutionVol,
		std::vector<float> &list)
{
	MultidimArray<double> resolutionVol_aux = resolutionVol;
	double init_res, last_res;

	init_res = list[0];
	last_res = list[(list.size()-1)];
	
	double last_resolution_2 = list[last_res];

	double lowest_res;
	lowest_res = list[1]; //Example resolutions between 10-300, list(0)=300, list(1)=290, it is used list(1) due to background
	//is at 300 and the smoothing cast values of 299 and they must be removed.

	// Count number of voxels with resolution
	size_t N=0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
		if ( (DIRECT_MULTIDIM_ELEM(resolutionVol, n)>=(last_resolution_2-0.001)) && (DIRECT_MULTIDIM_ELEM(resolutionVol, n)<=lowest_res) ) //the value 0.001 is a tolerance
			++N;

	// Get all resolution values
	MultidimArray<double> resolutions(N);
	size_t N_iter=0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
		if ( (DIRECT_MULTIDIM_ELEM(resolutionVol, n)>(last_resolution_2-0.001)) && (DIRECT_MULTIDIM_ELEM(resolutionVol, n)<=lowest_res))
		{
			DIRECT_MULTIDIM_ELEM(resolutions,N_iter++)=DIRECT_MULTIDIM_ELEM(resolutionVol, n);
		}

	//	median = resolutionVector[size_t(resolutionVector.size()*0.5)];

	// Sort value and get threshold
	std::sort(&A1D_ELEM(resolutions,0),&A1D_ELEM(resolutions,N));
	double medianResolution = A1D_ELEM(resolutions, (int)(0.5*N)); //median value

	std::cout << "median Resolution = " << medianResolution << std::endl;
}


void ProgMonoTomoFast::run()
{

	readAndPrepareData();

	Image<double> outputResolution;

 	

 	MultidimArray<int> &pMask = mask();
 	auto &pOutputResolution = outputResolution();

	pOutputResolution.resizeNoCopy(pMask);
 	pOutputResolution.initConstant(maxRes);

 	auto &pVfiltered = Vfiltered();
 	auto &pVresolutionFiltered = VresolutionFiltered();

	long Realelems = Xdim*Ydim*Zdim;
	float* amplitudeMS = new float[Realelems];
	float* amplitudeMN = new float[Realelems];

	double criticalZ=icdf_gauss(significance);
	double criticalW=-1;
	double resolution, last_resolution = 10000;  //A huge value for achieving
												//last_resolution < resolution
	float resolution_2;
	double freq, freqH, freqL;
	double max_meanS = -1e38;
	double cut_value = 0.025;
 	int boxsize = 50;

	double R_ = freq_step;

	if (R_<0.25)
		R_=0.25;

	double Nyquist = 2*sampling;
	if (minRes<2*sampling)
		minRes = Nyquist;

	bool doNextIteration=true;
	int last_fourier_idx = -1;

	int count_res = 0;
	FileName fnDebug;

	int iter=0;
	std::vector<float> list;

	std::cout << "Analyzing frequencies" << std::endl;
	std::cout << "                     " << std::endl;

	int xdim = XSIZE(pOutputResolution);
	int ydim = YSIZE(pOutputResolution);

	VRiesz = new float[Realelems];

	do
	{
		bool continueIter = false;
		bool breakIter = false;

		resolution2eval(count_res, R_,
						resolution, last_resolution,
						freq, freqH,
						last_fourier_idx, continueIter, breakIter);

		if (continueIter)
			continue;

		if (breakIter)
			break;

		std::cout << "resolution = " << resolution << std::endl;


		list.push_back(resolution);

		if (iter <2)
			resolution_2 = list[0];
		else
			resolution_2 = list[iter - 2];

		fnDebug = "Signal";

		freqL = freq + 0.01;

		amplitudeMonogenicSignal3D((float) freq, (float) freqH, (float) freqL, amplitudeMS, iter, fnDebug);

 		fnDebug = "Noise";
 		amplitudeMonogenicSignal3D((float) freq, (float) freqH, (float) freqL, amplitudeMN, iter, fnDebug);

 		Matrix2D<double> noiseMatrix;

 		Matrix2D<float> thresholdMatrix;
 		localNoise(amplitudeMN, noiseMatrix, boxsize, thresholdMatrix);


 		float sumS=0, sumS2=0, sumN=0, sumN2=0;
		long NN = 0, NS = 0;



		for (long n=0; n<Realelems; n++)
		{
			if (DIRECT_MULTIDIM_ELEM(pMask, n)>=1)
			{
				// double amplitudeValue=DIRECT_MULTIDIM_ELEM(amplitudeMS, n);
				// double amplitudeValueN=DIRECT_MULTIDIM_ELEM(amplitudeMN, n);
				// sumS  += amplitudeValue;
				// noiseValues.push_back(amplitudeValueN);
				// sumN  += amplitudeValueN;
				++NS;
				++NN;
			}
		}
	
// 		#ifdef DEBUG
// 		std::cout << "NS" << NS << std::endl;
// 		std::cout << "NVoxelsOriginalMask" << NVoxelsOriginalMask << std::endl;
// 		std::cout << "NS/NVoxelsOriginalMask = " << NS/NVoxelsOriginalMask << std::endl;
// 		#endif
		

			// if (NS == 0)
			// {
			// 	std::cout << "There are no points to compute inside the mask" << std::endl;
			// 	std::cout << "If the number of computed frequencies is low, perhaps the provided"
			// 			"mask is not enough tight to the volume, in that case please try another mask" << std::endl;
			// 	break;
			// }

			// double meanS=sumS/NS;

// 			#ifdef DEBUG
// 			  std::cout << "Iteration = " << iter << ",   Resolution= " << resolution <<
// 					  ",   Signal = " << meanS << ",   Noise = " << meanN << ",  Threshold = "
// 					  << thresholdNoise <<std::endl;
// 			#endif


			// if (meanS>max_meanS)
			// 	max_meanS = meanS;

			// if (meanS<0.001*max_meanS)
			// {
			// 	std::cout << "Search of resolutions stopped due to too low signal" << std::endl;
			// 	break;
			// }


			long n=0;
			for(size_t k=0; k<Zdim; ++k)
			{
				for(size_t i=0; i<Ydim; ++i)
				{
					for(size_t j=0; j<Xdim; ++j)
					{
						if (DIRECT_A3D_ELEM(pMask, k,i,j) >=1)
						{
							if (amplitudeMS[n] > MAT_ELEM(thresholdMatrix, i, j) )
							{

								DIRECT_A3D_ELEM(pMask,  k,i,j) = 1;
								DIRECT_A3D_ELEM(pOutputResolution, k,i,j) = resolution;
							}
							else{
								DIRECT_A3D_ELEM(pMask,  k,i,j) += 1;
								if (DIRECT_A3D_ELEM(pMask,  k,i,j) >2)
								{
									DIRECT_A3D_ELEM(pMask,  k,i,j) = -1;
									DIRECT_A3D_ELEM(pOutputResolution,  k,i,j) = resolution_2;
								}
							}
						}
						++n;
					}
				}
			}


// 			#ifdef DEBUG_MASK
// 			FileName fnmask_debug;
// 			fnmask_debug = formatString("maske_%i.vol", iter);
// 			mask.write(fnmask_debug);
// 			#endif


			if (doNextIteration)
			{
				if (resolution <= (minRes-0.001))
					doNextIteration = false;
			}

//		}
		iter++;
		last_resolution = resolution;
	} while (doNextIteration);

// 	Image<double> outputResolutionImage2;
// 	outputResolutionImage2() = pOutputResolution;
// 	outputResolutionImage2.write("resultado.vol");


	delete[] amplitudeMS;
	delete[] amplitudeMN;

	//Convolution with a real gaussian to get a smooth map
	MultidimArray<double> FilteredResolution = pOutputResolution;
	double sigma = 25.0;

	double resolutionThreshold;

	sigma = 3;

	realGaussianFilter(FilteredResolution, sigma);


	lowestResolutionbyPercentile(FilteredResolution, list, cut_value, resolutionThreshold);



	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FilteredResolution)
	{
		if ( (DIRECT_MULTIDIM_ELEM(FilteredResolution, n)<resolutionThreshold) && (DIRECT_MULTIDIM_ELEM(FilteredResolution, n)>DIRECT_MULTIDIM_ELEM(pOutputResolution, n)) )
			DIRECT_MULTIDIM_ELEM(FilteredResolution, n) = DIRECT_MULTIDIM_ELEM(pOutputResolution, n);
		if ( DIRECT_MULTIDIM_ELEM(FilteredResolution, n)<Nyquist)
			DIRECT_MULTIDIM_ELEM(FilteredResolution, n) = Nyquist;
	}

	realGaussianFilter(FilteredResolution, sigma);


	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FilteredResolution)
	{
		if ((DIRECT_MULTIDIM_ELEM(FilteredResolution, n)<resolutionThreshold)  && (DIRECT_MULTIDIM_ELEM(FilteredResolution, n)>DIRECT_MULTIDIM_ELEM(pOutputResolution, n)))
			DIRECT_MULTIDIM_ELEM(FilteredResolution, n) = DIRECT_MULTIDIM_ELEM(pOutputResolution, n);
		if ( DIRECT_MULTIDIM_ELEM(FilteredResolution, n)<Nyquist)
			DIRECT_MULTIDIM_ELEM(FilteredResolution, n) = Nyquist;
	}

	Image<double> outputResolutionImage;
	MultidimArray<double> resolutionFiltered;

	postProcessingLocalResolutions(FilteredResolution, list);

	outputResolutionImage() = FilteredResolution;
	outputResolutionImage.write(fnOut);

}
