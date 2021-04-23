/***************************************************************************
 *
 * Authors:     Jose Luis Vilas (joseluis.vilas-prieto@yale.edu)
 *                             or (jlvilas@cnb.csic.es)
 *              Hemant. D. Tagare (hemant.tagare@yale.edu)
 *
 * Yale University, New Haven, Connecticut, United States of America
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

#include "resolution_fso.h"
#include <core/metadata_extension.h>
#include <data/monogenic_signal.h>
#include <data/fourier_filter.h>
#include <random>
#include <chrono> 

#define SAVE_DIR_FSC

void ProgFSO::defineParams()
{
	addUsageLine("Calculate Fourier Shell Occupancy - FSO curve - via directional FSC measurements.");
	addUsageLine("Following outputs are generated:");
	addUsageLine("  1) FSO curve");
	addUsageLine("  2) Global resolution from FSO and FSC");
	addUsageLine("  3) 3DFSC");
	addUsageLine("  4) Anisotropic filter");
	addUsageLine("Reference: J.L. Vilas, H.D. Tagare, XXXXX (2021)");
	addUsageLine("+* Fourier Shell Occupancy (FSO)", true);
	addUsageLine("+ The Fourier Shell Occupancy Curve can be obtained from a set of directional FSC (see below).");
	addUsageLine("+ To do that, two half maps are used to determine the Global FSC at threshold 0.143. Then, the ratio between the number");
	addUsageLine("+ of directions with resolution higher (better) than the Global resolution and the total number of measured directions is");
	addUsageLine("+ calculated at different frequencies (resolutions). Note that this ratio is between 0 (resolution of all directions is worse than the global FSC)");
	addUsageLine("+ resolution than the global FSC)  and 1 (all directions present better resolution than the FSC) at a given resolution.");
	addUsageLine("+ In the particular case, FSO curve takes the value of 0.5 (FSO=0.5), then half of the directions are better, and.");
	addUsageLine("+ the other half are worse than the FSC, this situation occurs at the resoltuion of hte map. It means the FSO = 0.5 at the ");
	addUsageLine("+ FSC resolution. A map is isotropic if all directional resolution are similar, and anisotropic is there are significant resolution values along");
	addUsageLine("+ different directions. Thus, when the FSO presents a sharp cliff, a step-like function, the map will be isotropic.");
	addUsageLine("+ In contrast, when the OFSC shows a slope the map will be anisotropic. The lesser slope the higher resolution isotropy.");
	addUsageLine("+ ");
	addUsageLine("+* Directional Fourier Shell Correlation (dFSC)", true);
	addUsageLine("+ This program estimates the directional FSC between two half maps along all posible directions on the projection sphere.");
	addUsageLine("+ The directionality is measured by means of conical-like filters in Fourier Space. To avoid possible Gibbs effects ");
	addUsageLine("+ the filters are gaussian functions with their respective maxima along the filtering direction. A set of 321 directions ");
	addUsageLine("+ is used to cover the projection sphere, computing for each direction the directional FSC at 0.143 between the two half maps.");
	addUsageLine("+ The result is a set of 321 FSC curves (321 is the number of analyzed directions).");
	addUsageLine("+ The 3DFSC is then obtained from all curves by interpolation. Note that as well as it occurs with global FSC, the directional FSC is mask dependent.");
	addUsageLine(" ");
	addUsageLine("+* Resolution Distribution and 3DFSC", true);
	addUsageLine("+ The directional-FSC, dFSC is estimated along 321 directions on the projection sphere. For each direction the corresponding");
	addUsageLine("+ resolution is determined. Thus, it is possible to determine the resolution distribution on the projection sphere.");
	addUsageLine("+ This distribution is saved in the output metadata named resolutionDistribution.xmd. Also by means of all dFSC, the 3DFSC");
	addUsageLine("+ is calculated and saved as 3dFSC.mrc, which gives an idea about the information distributionin Fourier Space.");
	addUsageLine(" ");
	addUsageLine(" ");
	addSeeAlsoLine("resolution_fsc");

	addParamsLine("   --half1 <input_file>               : Input Half map 1");
	addParamsLine("   --half2 <input_file>               : Input Half map 2");

	addParamsLine("   [-o <output_folder=\"\">]          : Folder where the results will be stored.");

	addParamsLine("   [--sampling <Ts=1>]                : (Optical) Pixel size (Angstrom). If it is not provided by default will be 1 A/px.");
	addParamsLine("   [--mask <input_file=\"\">]         : (Optional) Smooth mask to remove noise. If it is not provided, the computation will be carried out without mask.");

	addParamsLine("   [--anglecone <ang_con=17>]         : (Optional) Angle Cone (angle between the axis and the  generatrix) for estimating the directional FSC");
	addParamsLine("   [--threshold <thrs=0.143>]		 : (Optional) Threshold for the FSC/directionalFSC estimation ");

    addParamsLine("   [--threedfsc_filter]           	 : (Optional) Put this flag to estimate the 3DFSC, and apply it as low pass filter to obtain a directionally filtered map. It mean to apply an anisotropic filter.");
	
	addParamsLine("   [--threads <Nthreads=1>]		     : (Optional) Number of threads to be used");

	addExampleLine("Resolution of two half maps half1.mrc and half2.mrc with a sampling rate of 2 A/px", false);
	addExampleLine("xmipp_resolution_fso --half1 half1.mrc  --half2 half2.mrc --sampling_rate 2 ");
	addExampleLine("Resolution of two half maps half1.mrc and half2.mrc with a sampling rate of 2 A/px and a mask mask.mrc", false);
	addExampleLine("xmipp_resolution_fso --half1 half1.mrc  --half2 half2.mrc --mask mask.mrc --sampling_rate 2 ");
}

void ProgFSO::readParams()
{
	fnhalf1 = getParam("--half1");
	fnhalf2 = getParam("--half2");
	fnOut = getParam("-o");

	sampling = getDoubleParam("--sampling");
	fnmask = getParam("--mask");
	ang_con = getDoubleParam("--anglecone");
	thrs = getDoubleParam("--threshold");
	do_3dfsc_filter = checkParam("--threedfsc_filter");
	
	Nthreads = getIntParam("--threads");
}


void ProgFSO::defineFrequencies(const MultidimArray< std::complex<double> > &mapfftV,
		const MultidimArray<double> &inputVol)
{
	// Initializing the frequency vectors
	freq_fourier_z.initZeros(ZSIZE(mapfftV));
	freq_fourier_x.initZeros(XSIZE(mapfftV));
	freq_fourier_y.initZeros(YSIZE(mapfftV));

	// u is the frequency
	double u;

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities
	VEC_ELEM(freq_fourier_z,0) = 1e-38;
	for(size_t k=1; k<ZSIZE(mapfftV); ++k){
		FFT_IDX2DIGFREQ(k,ZSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_z,k) = u;
	}

	VEC_ELEM(freq_fourier_y,0) = 1e-38;
	for(size_t k=1; k<YSIZE(mapfftV); ++k){
		FFT_IDX2DIGFREQ(k,YSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_y,k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = 1e-38;
	for(size_t k=1; k<XSIZE(mapfftV); ++k){
		FFT_IDX2DIGFREQ(k,XSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_x,k) = u;
	}

	//Initializing map with frequencies
	freqMap.resizeNoCopy(mapfftV);
	freqMap.initConstant(1.9);  //Nyquist is 2, we take 1.9 greater than Nyquist

	xvoldim = XSIZE(inputVol);
	yvoldim = YSIZE(inputVol);
	zvoldim = ZSIZE(inputVol);
	freqElems.initZeros(xvoldim/2+1);

	// Directional frequencies along each direction
	double uz, uy, ux, uz2, uz2y2;
	long n=0;
	int idx = 0;

	// Ncomps is the number of frequencies lesser than Nyquist
	long Ncomps = 0;
		
	for(size_t k=0; k<ZSIZE(mapfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz2 = uz*uz;
		for(size_t i=0; i<YSIZE(mapfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uz2y2 = uz2 + uy*uy;

			for(size_t j=0; j<XSIZE(mapfftV); ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				ux = sqrt(uz2y2 + ux*ux);

				if	(ux<=0.5)
				{
					idx = (int) round(ux * xvoldim);
					++Ncomps;
					DIRECT_MULTIDIM_ELEM(freqElems, idx) += 1;

					DIRECT_MULTIDIM_ELEM(freqMap,n) = 1/ux;

					if ((j == 0) && (uy<0))
					{
						DIRECT_MULTIDIM_ELEM(freqMap,n) = 1.9;
						DIRECT_MULTIDIM_ELEM(freqElems,idx) -= 1;
						--Ncomps;
					}
						
					if ((i == 0) && (j == 0) && (uz<0))
					{
						DIRECT_MULTIDIM_ELEM(freqMap,n) = 1.9;
						DIRECT_MULTIDIM_ELEM(freqElems,idx) -= 1;
						--Ncomps;
					}
						
				}				
				++n;
			}
		}
	}
	real_z1z2.initZeros(Ncomps);
}



void ProgFSO::arrangeFSC_and_fscGlobal(double sampling_rate,
				double &thrs, MultidimArray<double> &freq)
	{
		// cumpos is the the cumulative number of frequencies per shell number
		// First shell has 0 elements
		// second shell has the number of elements of the first shell
		// Third shell has the number of elements of the first+sencond shells and so on
		// freqElems in .h and set in defineFreq function
		cumpos.resizeNoCopy(NZYXSIZE(freqElems));

		DIRECT_MULTIDIM_ELEM(cumpos,0) = 0;
		for (long n = 1; n<NZYXSIZE(cumpos); ++n)
		{
			DIRECT_MULTIDIM_ELEM(cumpos,n) = DIRECT_MULTIDIM_ELEM(cumpos,n-1) + DIRECT_MULTIDIM_ELEM(freqElems,n-1);
		}
 		
		// real_z1z2, absz1_vec and absz2_vec will storage real(z1*conj(z2)), ||z1|| and ||z2|| using the
		//Fourier Coefficients z1, and z2 of both half maps. The storage will be in order, first elements of
		// the vector are the freq 0, next freq 1 and so on. Ncomps is the number of components with frequency
		// lesser than Nyquist (defined in defineFreq)

		absz1_vec = real_z1z2;
		absz2_vec = real_z1z2;
		
		// num and den are de numerator and denominator of the fsc
		MultidimArray<long> pos;
		MultidimArray<double> num, den1, den2, noiseAmp;
		num.initZeros(freqElems);
		pos.resizeNoCopy(num);

		freqidx.resizeNoCopy(real_z1z2);

		// arr2indx is the position of in Fourier space. It is defined in the .h
		// Because we only work with the frequencies lesser than nyquist their position
		// int he Fourier space is lost. arr2indx has the position of each frequency. 
		// It is used at the end of the algorithm to generate the 3DFSC
		arr2indx.resizeNoCopy(real_z1z2);
		arr2indx.initZeros();
		freqidx.initZeros();
		pos.initZeros();
		den1 = num;
		den2 = num;


		int ZdimFT1=(int)ZSIZE(FT1);
		int YdimFT1=(int)YSIZE(FT1);
		int XdimFT1=(int)XSIZE(FT1);

		// fx, fy, fz, defined in the .h are the frequencies of each pixel with frequency
		// lesser than Nyquist along each axis. They are used to compute the dot product
		// between each vector position and the direction of the cone to calculate the
		// directional FSC
		fx.resizeNoCopy(real_z1z2);
		fy.resizeNoCopy(real_z1z2);
		fz.resizeNoCopy(real_z1z2);

		long n = 0;
		for (int k=0; k<ZdimFT1; k++)
		{
			double uz = VEC_ELEM(freq_fourier_z, k);
			for (int i=0; i<YdimFT1; i++)
			{
				double uy = VEC_ELEM(freq_fourier_y, i);
				for (int j=0; j<XdimFT1; j++)
				{
					double ux = VEC_ELEM(freq_fourier_x, j);
					
					double iun = DIRECT_MULTIDIM_ELEM(freqMap,n);
					double f = 1/iun;
					++n;

					// Only reachable frequencies
					// To speed up the algorithm, only are considered those voxels with frequency lesser than Nyquist, 0.5. The vector idx_count
					// stores all frequencies lesser than Nyquist. This vector determines the frequency of each component of
					// real_z1z2, absz1_vec, absz2_vec.
					if (f>0.5)
						continue;
					
					// Index of each frequency
					int idx = (int) round(f * xvoldim);
					
					int idx_count = DIRECT_MULTIDIM_ELEM(cumpos, idx) + DIRECT_MULTIDIM_ELEM(pos, idx); 

					DIRECT_MULTIDIM_ELEM(arr2indx, idx_count) = n - 1;

					// Storing normalized frequencies
					DIRECT_MULTIDIM_ELEM(fx, idx_count) = (float) ux*iun;
					DIRECT_MULTIDIM_ELEM(fy, idx_count) = (float) uy*iun;
					DIRECT_MULTIDIM_ELEM(fz, idx_count) = (float) uz*iun;

					// In this vector we store the index of each Fourier coefficient, thus in the 
					// directional FSC estimation the calculus of the index is avoided speeding up the calculation
					DIRECT_MULTIDIM_ELEM(freqidx, idx_count) = idx;

					// Fourier coefficients of both halves
					std::complex<double> &z1 = dAkij(FT1, k, i, j);
					std::complex<double> &z2 = dAkij(FT2, k, i, j);

					double absz1 = abs(z1);
					double absz2 = abs(z2);

					DIRECT_MULTIDIM_ELEM(real_z1z2, idx_count) = (float) real(conj(z1)*z2);
					DIRECT_MULTIDIM_ELEM(absz1_vec, idx_count) = (float) absz1*absz1;
					DIRECT_MULTIDIM_ELEM(absz2_vec, idx_count) = (float) absz2*absz2;

					dAi(num,idx) += real(conj(z1) * z2);
					dAi(den1,idx) += absz1*absz1;
					dAi(den2,idx) += absz2*absz2;

					DIRECT_MULTIDIM_ELEM(pos, idx) +=1;
				}
			}
		}


		// The global FSC is stored as a metadata
		size_t id;
		MetaData mdRes;
		MultidimArray< double > frc;
		freq.initZeros(freqElems);
		frc.initZeros(freqElems);
		FOR_ALL_ELEMENTS_IN_ARRAY1D(freq)
		{
			dAi(frc,i) = dAi(num,i)/sqrt( (dAi(den1,i)) * (dAi(den2,i)) );
			dAi(freq,i) = (float) i / (xvoldim * sampling_rate);

			if (i>0)
			{
				id=mdRes.addObject();

				mdRes.setValue(MDL_RESOLUTION_FREQ,dAi(freq, i),id);
				mdRes.setValue(MDL_RESOLUTION_FRC,dAi(frc, i),id);
				mdRes.setValue(MDL_RESOLUTION_FREQREAL, 1./dAi(freq, i), id);
			}
		}
		mdRes.write(fnOut+"/GlobalFSC.xmd");

		

		// Here the FSC at 0.143 is obtained byt interpolating
		FOR_ALL_ELEMENTS_IN_ARRAY1D(freq)
		{
			double ff = dAi(frc,i);
			if ( (ff<=thrs) && (i>2) )
			{
				std::cout << std::endl << "shell = " << i << std::endl;
				fscshellNum = i;
				double y2, y1, x2, x1, slope, ny;
				y2 = ff;
				double ff_1 = dAi(freq, i-1);
				y1 = ff_1;
				x2 = ff;
				x1 = ff_1;

				slope = (y2 - y1)/(x2 - x1);
				ny = y2 - slope*x2;

				double fscResolution;
				fscResolution = 1/(slope*thrs + ny);
				std::cout << "Resolution " << fscResolution << std::endl;
				break;
			}
		}


		aniFilter.initZeros(freqidx);

	}



void ProgFSO::fscDir_fast(MultidimArray<double> &fsc, double rot, double tilt,
				MultidimArray<double> &threeD_FSC, MultidimArray<double> &normalizationMap,
				double &thrs, double &resol, size_t dirnumber)
{
	// FSCDIR_FAST: computes the directional FSC along a direction given by the angles rot and tilt.
	// Thus the directional resolution, resol, is estimated with the threshold, thrs.
	// The direcional FSC is stored in a metadata mdRes and in the multidimarray fsc. Later, this multidimarray
	// will be used to estimate the FSO.
	// In addition, the 3dfsc and a normalizationmap (needed to estimate the 3dfsc) are created. For each direction
	// these vectors are updated
	size_t dim = NZYXSIZE(freqElems);
	
	// numerator and denominator of the fsc
	MultidimArray<float> num, den1, den2;	
	num.initZeros(dim);
	den1.initZeros(dim);
	den2.initZeros(dim);

	// in vecidx the position n of the vector real_z1z2 is stored.
	// this allow to have a quick access to the frequencies and the
	// positions of the threeD_FSC without sweeping all positions
	std::vector<long> vecidx;
	// the used weight of the directional filter 
	std::vector<double> weightFSC3D;

	//Parameter to determine the direction of the cone
	float x_dir, y_dir, z_dir, cosAngle, aux;
	x_dir = sinf(tilt)*cosf(rot);
	y_dir = sinf(tilt)*sinf(rot);
	z_dir = cosf(tilt);

	

	cosAngle = (float) cos(ang_con);

	// It is multiply by 0.5 because later the weight is
	// cosine = sqrt(exp( -((cosine -1)*(cosine -1))*aux )); 
	// thus the computation of the weight is speeded up
	// aux = 4.0/((cos(ang_con) -1)*(cos(ang_con) -1));
	aux = (4.0/((cosAngle -1)*(cosAngle -1)));//*0.5;

	// Computing directional resolution
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(real_z1z2)
	{
		// frecuencies of the fourier position
		float ux = DIRECT_MULTIDIM_ELEM(fx, n);
		float uy = DIRECT_MULTIDIM_ELEM(fy, n);
		float uz = DIRECT_MULTIDIM_ELEM(fz, n);

		// angle between the position and the direction of the cone
		float cosine = fabs(x_dir*ux + y_dir*uy + z_dir*uz);
		
		// Only points inside the cone
		if (cosine >= cosAngle)	
		{
			// in fact it is sqrt(exp( -((cosine -1)*(cosine -1))*aux )); 
			// For sake of performance The sqrt is avoided dividing the by 2 in aux
			cosine = expf( -((cosine -1)*(cosine -1))*aux); 

			vecidx.push_back(n);
			// cosine *= cosine; Commented because is equivalent to remove the root square in aux
			weightFSC3D.push_back(cosine);

			// selecting the frequency of the shell
			size_t idxf = DIRECT_MULTIDIM_ELEM(freqidx, n);

			// computing the numerator and denominator
			dAi(num, idxf) += DIRECT_MULTIDIM_ELEM(real_z1z2, n)  * cosine;
			dAi(den1,idxf) += DIRECT_MULTIDIM_ELEM(absz1_vec, n)  * cosine;
			dAi(den2,idxf) += DIRECT_MULTIDIM_ELEM(absz2_vec, n)  * cosine;
		}
	}

	fsc.resizeNoCopy(num);
	fsc.initConstant(1.0);

	//The fsc is stored in a metadata and saved
	FileName fnmd;
	bool flagRes = true;
	size_t id;
	MetaData mdRes;
	FOR_ALL_ELEMENTS_IN_ARRAY1D(num)
	{
		double auxfsc = (dAi(num,i))/(sqrt(dAi(den1,i)*dAi(den2,i))+1e-38);
		dAi(fsc,i) = std::max(0.0, auxfsc);

		if (i>0)
		{
			#ifdef SAVE_DIR_FSC
			id=mdRes.addObject();
			double ff = (float) i / (xvoldim * sampling); // frequency
			mdRes.setValue(MDL_RESOLUTION_FREQ,ff,id);
			mdRes.setValue(MDL_RESOLUTION_FRC,dAi(fsc, i),id);
			mdRes.setValue(MDL_RESOLUTION_FREQREAL, 1./ff, id);
			#endif
			if ((i>2) && (dAi(fsc,i)<=thrs) && (flagRes))
			{
				flagRes = false;
				resol = 1./ff;
			}
		}
	}

	#ifdef SAVE_DIR_FSC
	fnmd = fnOut + formatString("/fscDirection_%i.xmd", dirnumber);
	mdRes.write(fnmd);
	#endif

	// the 3dfsc is computed and updated
	if (do_3dfsc_filter)
	{
		// sizevec is the number of points inside the cone
		size_t sizevec = vecidx.size();
		for (size_t kk = 0; kk< sizevec; ++kk)
		{
			double w = weightFSC3D[kk]; // the weights
			long n = vecidx[kk]; 		// position in fourier space
			size_t ind = DIRECT_MULTIDIM_ELEM(freqidx, n); 	// position in the fsc array
			
			// 3dfsc considering the weight
			DIRECT_MULTIDIM_ELEM(threeD_FSC, n) += w*DIRECT_MULTIDIM_ELEM(fsc, ind);
			// normalization map, we will divide by it once all direction are computed
			DIRECT_MULTIDIM_ELEM(normalizationMap, n) += w;
		}
	}
}


void ProgFSO::generateDirections(Matrix2D<float> &angles, bool alot)
{
	if (alot == true)
	{
		angles.initZeros(2,321);
		MAT_ELEM(angles,0,0)=0.0f;			MAT_ELEM(angles,1,0)=0.0f;
		MAT_ELEM(angles,0,1)=324.0f;		MAT_ELEM(angles,1,1)=63.4349f;
		MAT_ELEM(angles,0,2)=36.0f;			MAT_ELEM(angles,1,2)=63.4349f;
		MAT_ELEM(angles,0,3)=180.0f;		MAT_ELEM(angles,1,3)=63.435f;
		MAT_ELEM(angles,0,4)=252.0f;		MAT_ELEM(angles,1,4)=63.435f;
		MAT_ELEM(angles,0,5)=108.0f;		MAT_ELEM(angles,1,5)=63.435f;
		MAT_ELEM(angles,0,6)=324.0f;		MAT_ELEM(angles,1,6)=31.7175f;
		MAT_ELEM(angles,0,7)=36.0f;			MAT_ELEM(angles,1,7)=31.7175f;
		MAT_ELEM(angles,0,8)=0.0f;			MAT_ELEM(angles,1,8)=58.2825f;
		MAT_ELEM(angles,0,9)=288.0f;		MAT_ELEM(angles,1,9)=58.2825f;
		MAT_ELEM(angles,0,10)=342.0f;		MAT_ELEM(angles,1,10)=90.0f;
		MAT_ELEM(angles,0,11)=306.0f;		MAT_ELEM(angles,1,11)=90.0f;
		MAT_ELEM(angles,0,12)=72.0f;		MAT_ELEM(angles,1,12)=58.2825f;
		MAT_ELEM(angles,0,13)=18.0f;		MAT_ELEM(angles,1,13)=90.0f;
		MAT_ELEM(angles,0,14)=54.0f;		MAT_ELEM(angles,1,14)=90.0f;
		MAT_ELEM(angles,0,15)=90.0f;		MAT_ELEM(angles,1,15)=90.0f;
		MAT_ELEM(angles,0,16)=216.0f;		MAT_ELEM(angles,1,16)=58.282f;
		MAT_ELEM(angles,0,17)=144.0f;		MAT_ELEM(angles,1,17)=58.282f;
		MAT_ELEM(angles,0,18)=180.0f;		MAT_ELEM(angles,1,18)=31.718f;
		MAT_ELEM(angles,0,19)=252.0f;		MAT_ELEM(angles,1,19)=31.718f;
		MAT_ELEM(angles,0,20)=108.0f;		MAT_ELEM(angles,1,20)=31.718f;
		MAT_ELEM(angles,0,21)=346.3862f;	MAT_ELEM(angles,1,21)=43.6469f;
		MAT_ELEM(angles,0,22)=58.3862f;		MAT_ELEM(angles,1,22)=43.6469f;
		MAT_ELEM(angles,0,23)=274.3862f;	MAT_ELEM(angles,1,23)=43.6469f;
		MAT_ELEM(angles,0,24)=0.0f;			MAT_ELEM(angles,1,24)=90.0f;
		MAT_ELEM(angles,0,25)=72.0f;		MAT_ELEM(angles,1,25)=90.0f;
		MAT_ELEM(angles,0,26)=288.0f;		MAT_ELEM(angles,1,26)=90.0f;
		MAT_ELEM(angles,0,27)=225.7323f;	MAT_ELEM(angles,1,27)=73.955f;
		MAT_ELEM(angles,0,28)=153.7323f;	MAT_ELEM(angles,1,28)=73.955f;
		MAT_ELEM(angles,0,29)=216.0f;		MAT_ELEM(angles,1,29)=26.565f;
		MAT_ELEM(angles,0,30)=144.0f;		MAT_ELEM(angles,1,30)=26.565f;
		MAT_ELEM(angles,0,31)=0.0f;			MAT_ELEM(angles,1,31)=26.5651f;
		MAT_ELEM(angles,0,32)=72.0f;		MAT_ELEM(angles,1,32)=26.5651f;
		MAT_ELEM(angles,0,33)=288.0f;		MAT_ELEM(angles,1,33)=26.5651f;
		MAT_ELEM(angles,0,34)=350.2677f;	MAT_ELEM(angles,1,34)=73.9549f;
		MAT_ELEM(angles,0,35)=62.2677f;		MAT_ELEM(angles,1,35)=73.9549f;
		MAT_ELEM(angles,0,36)=278.2677f;	MAT_ELEM(angles,1,36)=73.9549f;
		MAT_ELEM(angles,0,37)=206.2677f;	MAT_ELEM(angles,1,37)=73.955f;
		MAT_ELEM(angles,0,38)=134.2677f;	MAT_ELEM(angles,1,38)=73.955f;
		MAT_ELEM(angles,0,39)=202.3862f;	MAT_ELEM(angles,1,39)=43.647f;
		MAT_ELEM(angles,0,40)=130.3862f;	MAT_ELEM(angles,1,40)=43.647f;
		MAT_ELEM(angles,0,41)=13.6138f;		MAT_ELEM(angles,1,41)=43.6469f;
		MAT_ELEM(angles,0,42)=85.6138f;		MAT_ELEM(angles,1,42)=43.6469f;
		MAT_ELEM(angles,0,43)=301.6138f;	MAT_ELEM(angles,1,43)=43.6469f;
		MAT_ELEM(angles,0,44)=9.7323f;		MAT_ELEM(angles,1,44)=73.9549f;
		MAT_ELEM(angles,0,45)=81.7323f;		MAT_ELEM(angles,1,45)=73.9549f;
		MAT_ELEM(angles,0,46)=297.7323f;	MAT_ELEM(angles,1,46)=73.9549f;
		MAT_ELEM(angles,0,47)=36.0f;		MAT_ELEM(angles,1,47)=90.0f;
		MAT_ELEM(angles,0,48)=324.0f;		MAT_ELEM(angles,1,48)=90.0f;
		MAT_ELEM(angles,0,49)=229.6138f;	MAT_ELEM(angles,1,49)=43.647f;
		MAT_ELEM(angles,0,50)=157.6138f;	MAT_ELEM(angles,1,50)=43.647f;
		MAT_ELEM(angles,0,51)=324.0f;		MAT_ELEM(angles,1,51)=15.8587f;
		MAT_ELEM(angles,0,52)=36.0f;		MAT_ELEM(angles,1,52)=15.8587f;
		MAT_ELEM(angles,0,53)=341.533;	MAT_ELEM(angles,1,53)=59.6208;
		MAT_ELEM(angles,0,54)=306.467;	MAT_ELEM(angles,1,54)=59.6208;
		MAT_ELEM(angles,0,55)=333.5057;	MAT_ELEM(angles,1,55)=76.5584;
		MAT_ELEM(angles,0,56)=314.4943;	MAT_ELEM(angles,1,56)=76.5584;
		MAT_ELEM(angles,0,57)=53.533;	MAT_ELEM(angles,1,57)=59.6208;
		MAT_ELEM(angles,0,58)=26.4943;	MAT_ELEM(angles,1,58)=76.5584;
		MAT_ELEM(angles,0,59)=45.5057;	MAT_ELEM(angles,1,59)=76.5584;
		MAT_ELEM(angles,0,60)=197.533;	MAT_ELEM(angles,1,60)=59.621;
		MAT_ELEM(angles,0,61)=162.467;	MAT_ELEM(angles,1,61)=59.621;
		MAT_ELEM(angles,0,62)=180;		MAT_ELEM(angles,1,62)=47.576;
		MAT_ELEM(angles,0,63)=269.533;	MAT_ELEM(angles,1,63)=59.621;
		MAT_ELEM(angles,0,64)=252;		MAT_ELEM(angles,1,64)=47.576;
		MAT_ELEM(angles,0,65)=108;		MAT_ELEM(angles,1,65)=47.576;
		MAT_ELEM(angles,0,66)=324;		MAT_ELEM(angles,1,66)=47.5762;
		MAT_ELEM(angles,0,67)=36;		MAT_ELEM(angles,1,67)=47.5762;
		MAT_ELEM(angles,0,68)=18.467;	MAT_ELEM(angles,1,68)=59.6208;
		MAT_ELEM(angles,0,69)=170.4943;	MAT_ELEM(angles,1,69)=76.558;
		MAT_ELEM(angles,0,70)=117.5057;	MAT_ELEM(angles,1,70)=76.558;
		MAT_ELEM(angles,0,71)=189.5057;	MAT_ELEM(angles,1,71)=76.558;
		MAT_ELEM(angles,0,72)=242.4943;	MAT_ELEM(angles,1,72)=76.558;
		MAT_ELEM(angles,0,73)=261.5057;	MAT_ELEM(angles,1,73)=76.558;
		MAT_ELEM(angles,0,74)=98.4943;	MAT_ELEM(angles,1,74)=76.558;
		MAT_ELEM(angles,0,75)=234.467;	MAT_ELEM(angles,1,75)=59.621;
		MAT_ELEM(angles,0,76)=125.533;	MAT_ELEM(angles,1,76)=59.621;
		MAT_ELEM(angles,0,77)=180;		MAT_ELEM(angles,1,77)=15.859;
		MAT_ELEM(angles,0,78)=252;		MAT_ELEM(angles,1,78)=15.859;
		MAT_ELEM(angles,0,79)=90.467;	MAT_ELEM(angles,1,79)=59.621;
		MAT_ELEM(angles,0,80)=108;		MAT_ELEM(angles,1,80)=15.859;
		MAT_ELEM(angles,0,81)=0;		MAT_ELEM(angles,1,81)=42.8321;
		MAT_ELEM(angles,0,82)=72;		MAT_ELEM(angles,1,82)=42.8321;
		MAT_ELEM(angles,0,83)=288;		MAT_ELEM(angles,1,83)=42.8321;
		MAT_ELEM(angles,0,84)=4.7693;	MAT_ELEM(angles,1,84)=81.9488;
		MAT_ELEM(angles,0,85)=76.7693;	MAT_ELEM(angles,1,85)=81.9488;
		MAT_ELEM(angles,0,86)=292.7693;	MAT_ELEM(angles,1,86)=81.9488;
		MAT_ELEM(angles,0,87)=220.7693;	MAT_ELEM(angles,1,87)=81.9488;
		MAT_ELEM(angles,0,88)=148.7693;	MAT_ELEM(angles,1,88)=81.9488;
		MAT_ELEM(angles,0,89)=224.2677;	MAT_ELEM(angles,1,89)=34.924;
		MAT_ELEM(angles,0,90)=152.2677;	MAT_ELEM(angles,1,90)=34.924;
		MAT_ELEM(angles,0,91)=13.5146;	MAT_ELEM(angles,1,91)=20.3172;
		MAT_ELEM(angles,0,92)=85.5146;	MAT_ELEM(angles,1,92)=20.3172;
		MAT_ELEM(angles,0,93)=301.5146;	MAT_ELEM(angles,1,93)=20.3172;
		MAT_ELEM(angles,0,94)=346.1363;	MAT_ELEM(angles,1,94)=66.7276;
		MAT_ELEM(angles,0,95)=58.1363;	MAT_ELEM(angles,1,95)=66.7276;
		MAT_ELEM(angles,0,96)=274.1363;	MAT_ELEM(angles,1,96)=66.7276;
		MAT_ELEM(angles,0,97)=197.8362;	MAT_ELEM(angles,1,97)=75.105;
		MAT_ELEM(angles,0,98)=269.8362;	MAT_ELEM(angles,1,98)=75.105;
		MAT_ELEM(angles,0,99)=125.8362;	MAT_ELEM(angles,1,99)=75.105;
		MAT_ELEM(angles,0,100)=199.6899;MAT_ELEM(angles,1,100)=51.609;
		MAT_ELEM(angles,0,101)=127.6899;MAT_ELEM(angles,1,101)=51.609;
		MAT_ELEM(angles,0,102)=334.8124;MAT_ELEM(angles,1,102)=45.0621;
		MAT_ELEM(angles,0,103)=46.8124;	MAT_ELEM(angles,1,103)=45.0621;
		MAT_ELEM(angles,0,104)=175.3133;MAT_ELEM(angles,1,104)=83.2562;
		MAT_ELEM(angles,0,105)=247.3133;MAT_ELEM(angles,1,105)=83.2562;
		MAT_ELEM(angles,0,106)=103.3133;MAT_ELEM(angles,1,106)=83.2562;
		MAT_ELEM(angles,0,107)=229.8637;MAT_ELEM(angles,1,107)=66.728;
		MAT_ELEM(angles,0,108)=157.8637;MAT_ELEM(angles,1,108)=66.728;
		MAT_ELEM(angles,0,109)=202.4854;MAT_ELEM(angles,1,109)=20.317;
		MAT_ELEM(angles,0,110)=130.4854;MAT_ELEM(angles,1,110)=20.317;
		MAT_ELEM(angles,0,111)=16.3101;	MAT_ELEM(angles,1,111)=51.6091;
		MAT_ELEM(angles,0,112)=88.3101;	MAT_ELEM(angles,1,112)=51.6091;
		MAT_ELEM(angles,0,113)=304.3101;MAT_ELEM(angles,1,113)=51.6091;
		MAT_ELEM(angles,0,114)=18.1638;	MAT_ELEM(angles,1,114)=75.1046;
		MAT_ELEM(angles,0,115)=306.1638;MAT_ELEM(angles,1,115)=75.1046;
		MAT_ELEM(angles,0,116)=40.6867;	MAT_ELEM(angles,1,116)=83.2562;
		MAT_ELEM(angles,0,117)=328.6867;MAT_ELEM(angles,1,117)=83.2562;
		MAT_ELEM(angles,0,118)=241.1876;MAT_ELEM(angles,1,118)=45.062;
		MAT_ELEM(angles,0,119)=97.1876;	MAT_ELEM(angles,1,119)=45.062;
		MAT_ELEM(angles,0,120)=169.1876;MAT_ELEM(angles,1,120)=45.062;
		MAT_ELEM(angles,0,121)=351.7323;MAT_ELEM(angles,1,121)=34.9243;
		MAT_ELEM(angles,0,122)=63.7323;	MAT_ELEM(angles,1,122)=34.9243;
		MAT_ELEM(angles,0,123)=279.7323;MAT_ELEM(angles,1,123)=34.9243;
		MAT_ELEM(angles,0,124)=355.2307;MAT_ELEM(angles,1,124)=81.9488;
		MAT_ELEM(angles,0,125)=67.2307;	MAT_ELEM(angles,1,125)=81.9488;
		MAT_ELEM(angles,0,126)=283.2307;MAT_ELEM(angles,1,126)=81.9488;
		MAT_ELEM(angles,0,127)=216;		MAT_ELEM(angles,1,127)=73.733;
		MAT_ELEM(angles,0,128)=144;		MAT_ELEM(angles,1,128)=73.733;
		MAT_ELEM(angles,0,129)=207.7323;MAT_ELEM(angles,1,129)=34.924;
		MAT_ELEM(angles,0,130)=135.7323;MAT_ELEM(angles,1,130)=34.924;
		MAT_ELEM(angles,0,131)=346.4854;MAT_ELEM(angles,1,131)=20.3172;
		MAT_ELEM(angles,0,132)=58.4854;	MAT_ELEM(angles,1,132)=20.3172;
		MAT_ELEM(angles,0,133)=274.4854;MAT_ELEM(angles,1,133)=20.3172;
		MAT_ELEM(angles,0,134)=341.8362;MAT_ELEM(angles,1,134)=75.1046;
		MAT_ELEM(angles,0,135)=53.8362;	MAT_ELEM(angles,1,135)=75.1046;
		MAT_ELEM(angles,0,136)=202.1363;MAT_ELEM(angles,1,136)=66.728;
		MAT_ELEM(angles,0,137)=130.1363;MAT_ELEM(angles,1,137)=66.728;
		MAT_ELEM(angles,0,138)=190.8124;MAT_ELEM(angles,1,138)=45.062;
		MAT_ELEM(angles,0,139)=262.8124;MAT_ELEM(angles,1,139)=45.062;
		MAT_ELEM(angles,0,140)=118.8124;MAT_ELEM(angles,1,140)=45.062;
		MAT_ELEM(angles,0,141)=343.6899;MAT_ELEM(angles,1,141)=51.6091;
		MAT_ELEM(angles,0,142)=55.6899;	MAT_ELEM(angles,1,142)=51.6091;
		MAT_ELEM(angles,0,143)=271.6899;MAT_ELEM(angles,1,143)=51.6091;
		MAT_ELEM(angles,0,144)=184.6867;MAT_ELEM(angles,1,144)=83.2562;
		MAT_ELEM(angles,0,145)=256.6867;MAT_ELEM(angles,1,145)=83.2562;
		MAT_ELEM(angles,0,146)=112.6867;MAT_ELEM(angles,1,146)=83.2562;
		MAT_ELEM(angles,0,147)=234.1638;MAT_ELEM(angles,1,147)=75.105;
		MAT_ELEM(angles,0,148)=90.1638;	MAT_ELEM(angles,1,148)=75.105;
		MAT_ELEM(angles,0,149)=162.1638;MAT_ELEM(angles,1,149)=75.105;
		MAT_ELEM(angles,0,150)=229.5146;MAT_ELEM(angles,1,150)=20.317;
		MAT_ELEM(angles,0,151)=157.5146;MAT_ELEM(angles,1,151)=20.317;
		MAT_ELEM(angles,0,152)=25.1876;	MAT_ELEM(angles,1,152)=45.0621;
		MAT_ELEM(angles,0,153)=313.1876;MAT_ELEM(angles,1,153)=45.0621;
		MAT_ELEM(angles,0,154)=13.8637;	MAT_ELEM(angles,1,154)=66.7276;
		MAT_ELEM(angles,0,155)=85.8637;	MAT_ELEM(angles,1,155)=66.7276;
		MAT_ELEM(angles,0,156)=301.8637;MAT_ELEM(angles,1,156)=66.7276;
		MAT_ELEM(angles,0,157)=31.3133;	MAT_ELEM(angles,1,157)=83.2562;
		MAT_ELEM(angles,0,158)=319.3133;MAT_ELEM(angles,1,158)=83.2562;
		MAT_ELEM(angles,0,159)=232.3101;MAT_ELEM(angles,1,159)=51.609;
		MAT_ELEM(angles,0,160)=160.3101;MAT_ELEM(angles,1,160)=51.609;
		MAT_ELEM(angles,0,161)=8.2677;	MAT_ELEM(angles,1,161)=34.9243;
		MAT_ELEM(angles,0,162)=80.2677;	MAT_ELEM(angles,1,162)=34.9243;
		MAT_ELEM(angles,0,163)=296.2677;MAT_ELEM(angles,1,163)=34.9243;
		MAT_ELEM(angles,0,164)=0;		MAT_ELEM(angles,1,164)=73.733;
		MAT_ELEM(angles,0,165)=72;		MAT_ELEM(angles,1,165)=73.733;
		MAT_ELEM(angles,0,166)=288;		MAT_ELEM(angles,1,166)=73.733;
		MAT_ELEM(angles,0,167)=211.2307;MAT_ELEM(angles,1,167)=81.9488;
		MAT_ELEM(angles,0,168)=139.2307;MAT_ELEM(angles,1,168)=81.9488;
		MAT_ELEM(angles,0,169)=216;		MAT_ELEM(angles,1,169)=42.832;
		MAT_ELEM(angles,0,170)=144;		MAT_ELEM(angles,1,170)=42.832;
		MAT_ELEM(angles,0,171)=0;		MAT_ELEM(angles,1,171)=12.9432;
		MAT_ELEM(angles,0,172)=72;		MAT_ELEM(angles,1,172)=12.9432;
		MAT_ELEM(angles,0,173)=288;		MAT_ELEM(angles,1,173)=12.9432;
		MAT_ELEM(angles,0,174)=337.2786;MAT_ELEM(angles,1,174)=68.041;
		MAT_ELEM(angles,0,175)=49.2786; MAT_ELEM(angles,1,175)=68.041;
		MAT_ELEM(angles,0,176)=193.2786;MAT_ELEM(angles,1,176)=68.041;
		MAT_ELEM(angles,0,177)=265.2786;MAT_ELEM(angles,1,177)=68.041;
		MAT_ELEM(angles,0,178)=121.2786;MAT_ELEM(angles,1,178)=68.041;
		MAT_ELEM(angles,0,179)=189.4537;MAT_ELEM(angles,1,179)=53.278;
		MAT_ELEM(angles,0,180)=261.4537;MAT_ELEM(angles,1,180)=53.278;
		MAT_ELEM(angles,0,181)=117.4537;MAT_ELEM(angles,1,181)=53.278;
		MAT_ELEM(angles,0,182)=333.4537;MAT_ELEM(angles,1,182)=53.2783;
		MAT_ELEM(angles,0,183)=45.4537; MAT_ELEM(angles,1,183)=53.2783;
		MAT_ELEM(angles,0,184)=180;		MAT_ELEM(angles,1,184)=76.378;
		MAT_ELEM(angles,0,185)=252;		MAT_ELEM(angles,1,185)=76.378;
		MAT_ELEM(angles,0,186)=108;		MAT_ELEM(angles,1,186)=76.378;
		MAT_ELEM(angles,0,187)=238.7214;MAT_ELEM(angles,1,187)=68.041;
		MAT_ELEM(angles,0,188)=94.7214;	MAT_ELEM(angles,1,188)=68.041;
		MAT_ELEM(angles,0,189)=166.7214;MAT_ELEM(angles,1,189)=68.041;
		MAT_ELEM(angles,0,190)=216;		MAT_ELEM(angles,1,190)=12.943;
		MAT_ELEM(angles,0,191)=144;		MAT_ELEM(angles,1,191)=12.943;
		MAT_ELEM(angles,0,192)=26.5463;	MAT_ELEM(angles,1,192)=53.2783;
		MAT_ELEM(angles,0,193)=314.5463;MAT_ELEM(angles,1,193)=53.2783;
		MAT_ELEM(angles,0,194)=22.7214;	MAT_ELEM(angles,1,194)=68.041;
		MAT_ELEM(angles,0,195)=310.7214;MAT_ELEM(angles,1,195)=68.041;
		MAT_ELEM(angles,0,196)=36;		MAT_ELEM(angles,1,196)=76.3782;
		MAT_ELEM(angles,0,197)=324;		MAT_ELEM(angles,1,197)=76.3782;
		MAT_ELEM(angles,0,198)=242.5463;MAT_ELEM(angles,1,198)=53.278;
		MAT_ELEM(angles,0,199)=98.5463;	MAT_ELEM(angles,1,199)=53.278;
		MAT_ELEM(angles,0,200)=170.5463;MAT_ELEM(angles,1,200)=53.278;
		MAT_ELEM(angles,0,201)=336.7264;MAT_ELEM(angles,1,201)=37.1611;
		MAT_ELEM(angles,0,202)=48.7264;	MAT_ELEM(angles,1,202)=37.1611;
		MAT_ELEM(angles,0,203)=351;		MAT_ELEM(angles,1,203)=90;
		MAT_ELEM(angles,0,204)=63;		MAT_ELEM(angles,1,204)=90;
		MAT_ELEM(angles,0,205)=279;		MAT_ELEM(angles,1,205)=90;
		MAT_ELEM(angles,0,206)=221.1634;MAT_ELEM(angles,1,206)=66.042;
		MAT_ELEM(angles,0,207)=149.1634;MAT_ELEM(angles,1,207)=66.042;
		MAT_ELEM(angles,0,208)=196.498;	MAT_ELEM(angles,1,208)=27.943;
		MAT_ELEM(angles,0,209)=268.498;	MAT_ELEM(angles,1,209)=27.943;
		MAT_ELEM(angles,0,210)=124.498;	MAT_ELEM(angles,1,210)=27.943;
		MAT_ELEM(angles,0,211)=340.498;	MAT_ELEM(angles,1,211)=27.9429;
		MAT_ELEM(angles,0,212)=52.498;	MAT_ELEM(angles,1,212)=27.9429;
		MAT_ELEM(angles,0,213)=346.0516;MAT_ELEM(angles,1,213)=81.9568;
		MAT_ELEM(angles,0,214)=58.0516;	MAT_ELEM(angles,1,214)=81.9568;
		MAT_ELEM(angles,0,215)=274.0516;MAT_ELEM(angles,1,215)=81.9568;
		MAT_ELEM(angles,0,216)=210.8366;MAT_ELEM(angles,1,216)=66.042;
		MAT_ELEM(angles,0,217)=138.8366;MAT_ELEM(angles,1,217)=66.042;
		MAT_ELEM(angles,0,218)=192.7264;MAT_ELEM(angles,1,218)=37.161;
		MAT_ELEM(angles,0,219)=264.7264;MAT_ELEM(angles,1,219)=37.161;
		MAT_ELEM(angles,0,220)=120.7264;MAT_ELEM(angles,1,220)=37.161;
		MAT_ELEM(angles,0,221)=6.0948;	MAT_ELEM(angles,1,221)=50.7685;
		MAT_ELEM(angles,0,222)=78.0948;	MAT_ELEM(angles,1,222)=50.7685;
		MAT_ELEM(angles,0,223)=294.0948;MAT_ELEM(angles,1,223)=50.7685;
		MAT_ELEM(angles,0,224)=13.9484;	MAT_ELEM(angles,1,224)=81.9568;
		MAT_ELEM(angles,0,225)=85.9484;	MAT_ELEM(angles,1,225)=81.9568;
		MAT_ELEM(angles,0,226)=301.9484;MAT_ELEM(angles,1,226)=81.9568;
		MAT_ELEM(angles,0,227)=45;		MAT_ELEM(angles,1,227)=90;
		MAT_ELEM(angles,0,228)=333;		MAT_ELEM(angles,1,228)=90;
		MAT_ELEM(angles,0,229)=239.2736;MAT_ELEM(angles,1,229)=37.161;
		MAT_ELEM(angles,0,230)=95.2736;	MAT_ELEM(angles,1,230)=37.161;
		MAT_ELEM(angles,0,231)=167.2736;MAT_ELEM(angles,1,231)=37.161;
		MAT_ELEM(angles,0,232)=324;		MAT_ELEM(angles,1,232)=7.9294;
		MAT_ELEM(angles,0,233)=36;		MAT_ELEM(angles,1,233)=7.9294;
		MAT_ELEM(angles,0,234)=332.6069;MAT_ELEM(angles,1,234)=61.2449;
		MAT_ELEM(angles,0,235)=315.3931;MAT_ELEM(angles,1,235)=61.2449;
		MAT_ELEM(angles,0,236)=328.9523;MAT_ELEM(angles,1,236)=69.9333;
		MAT_ELEM(angles,0,237)=319.0477;MAT_ELEM(angles,1,237)=69.9333;
		MAT_ELEM(angles,0,238)=44.6069;	MAT_ELEM(angles,1,238)=61.2449;
		MAT_ELEM(angles,0,239)=31.0477;	MAT_ELEM(angles,1,239)=69.9333;
		MAT_ELEM(angles,0,240)=40.9523;	MAT_ELEM(angles,1,240)=69.9333;
		MAT_ELEM(angles,0,241)=188.6069;MAT_ELEM(angles,1,241)=61.245;
		MAT_ELEM(angles,0,242)=171.3931;MAT_ELEM(angles,1,242)=61.245;
		MAT_ELEM(angles,0,243)=180;		MAT_ELEM(angles,1,243)=55.506;
		MAT_ELEM(angles,0,244)=260.6069;MAT_ELEM(angles,1,244)=61.245;
		MAT_ELEM(angles,0,245)=252;		MAT_ELEM(angles,1,245)=55.506;
		MAT_ELEM(angles,0,246)=108;		MAT_ELEM(angles,1,246)=55.506;
		MAT_ELEM(angles,0,247)=324;		MAT_ELEM(angles,1,247)=39.6468;
		MAT_ELEM(angles,0,248)=36;		MAT_ELEM(angles,1,248)=39.6468;
		MAT_ELEM(angles,0,249)=9.299;	MAT_ELEM(angles,1,249)=58.6205;
		MAT_ELEM(angles,0,250)=278.701;	MAT_ELEM(angles,1,250)=58.6205;
		MAT_ELEM(angles,0,251)=166.1881;MAT_ELEM(angles,1,251)=83.2609;
		MAT_ELEM(angles,0,252)=121.8119;MAT_ELEM(angles,1,252)=83.2609;
		MAT_ELEM(angles,0,253)=81.299;	MAT_ELEM(angles,1,253)=58.6205;
		MAT_ELEM(angles,0,254)=193.8119;MAT_ELEM(angles,1,254)=83.2609;
		MAT_ELEM(angles,0,255)=238.1881;MAT_ELEM(angles,1,255)=83.2609;
		MAT_ELEM(angles,0,256)=265.8119;MAT_ELEM(angles,1,256)=83.2609;
		MAT_ELEM(angles,0,257)=94.1881;	MAT_ELEM(angles,1,257)=83.2609;
		MAT_ELEM(angles,0,258)=225.299;	MAT_ELEM(angles,1,258)=58.621;
		MAT_ELEM(angles,0,259)=134.701;	MAT_ELEM(angles,1,259)=58.621;
		MAT_ELEM(angles,0,260)=180;		MAT_ELEM(angles,1,260)=23.788;
		MAT_ELEM(angles,0,261)=252;		MAT_ELEM(angles,1,261)=23.788;
		MAT_ELEM(angles,0,262)=108;		MAT_ELEM(angles,1,262)=23.788;
		MAT_ELEM(angles,0,263)=353.9052;MAT_ELEM(angles,1,263)=50.7685;
		MAT_ELEM(angles,0,264)=65.9052; MAT_ELEM(angles,1,264)=50.7685;
		MAT_ELEM(angles,0,265)=281.9052;MAT_ELEM(angles,1,265)=50.7685;
		MAT_ELEM(angles,0,266)=9;		MAT_ELEM(angles,1,266)=90;
		MAT_ELEM(angles,0,267)=81;		MAT_ELEM(angles,1,267)=90;
		MAT_ELEM(angles,0,268)=297;		MAT_ELEM(angles,1,268)=90;
		MAT_ELEM(angles,0,269)=229.9484;MAT_ELEM(angles,1,269)=81.9568;
		MAT_ELEM(angles,0,270)=157.9484;MAT_ELEM(angles,1,270)=81.9568;
		MAT_ELEM(angles,0,271)=235.502;	MAT_ELEM(angles,1,271)=27.943;
		MAT_ELEM(angles,0,272)=91.502;	MAT_ELEM(angles,1,272)=27.943;
		MAT_ELEM(angles,0,273)=163.502;	MAT_ELEM(angles,1,273)=27.943;
		MAT_ELEM(angles,0,274)=19.502;	MAT_ELEM(angles,1,274)=27.9429;
		MAT_ELEM(angles,0,275)=307.502;	MAT_ELEM(angles,1,275)=27.9429;
		MAT_ELEM(angles,0,276)=354.8366;MAT_ELEM(angles,1,276)=66.0423;
		MAT_ELEM(angles,0,277)=66.8366;	MAT_ELEM(angles,1,277)=66.0423;
		MAT_ELEM(angles,0,278)=282.8366;MAT_ELEM(angles,1,278)=66.0423;
		MAT_ELEM(angles,0,279)=202.0516;MAT_ELEM(angles,1,279)=81.9568;
		MAT_ELEM(angles,0,280)=130.0516;MAT_ELEM(angles,1,280)=81.9568;
		MAT_ELEM(angles,0,281)=209.9052;MAT_ELEM(angles,1,281)=50.768;
		MAT_ELEM(angles,0,282)=137.9052;MAT_ELEM(angles,1,282)=50.768;
		MAT_ELEM(angles,0,283)=23.2736;	MAT_ELEM(angles,1,283)=37.1611;
		MAT_ELEM(angles,0,284)=311.2736;MAT_ELEM(angles,1,284)=37.1611;
		MAT_ELEM(angles,0,285)=5.1634;	MAT_ELEM(angles,1,285)=66.0423;
		MAT_ELEM(angles,0,286)=77.1634;	MAT_ELEM(angles,1,286)=66.0423;
		MAT_ELEM(angles,0,287)=293.1634;MAT_ELEM(angles,1,287)=66.0423;
		MAT_ELEM(angles,0,288)=27;		MAT_ELEM(angles,1,288)=90;
		MAT_ELEM(angles,0,289)=315;		MAT_ELEM(angles,1,289)=90;
		MAT_ELEM(angles,0,290)=222.0948;MAT_ELEM(angles,1,290)=50.768;
		MAT_ELEM(angles,0,291)=150.0948;MAT_ELEM(angles,1,291)=50.768;
		MAT_ELEM(angles,0,292)=324;		MAT_ELEM(angles,1,292)=23.7881;
		MAT_ELEM(angles,0,293)=36;		MAT_ELEM(angles,1,293)=23.7881;
		MAT_ELEM(angles,0,294)=350.701;	MAT_ELEM(angles,1,294)=58.6205;
		MAT_ELEM(angles,0,295)=297.299;	MAT_ELEM(angles,1,295)=58.6205;
		MAT_ELEM(angles,0,296)=337.8119;MAT_ELEM(angles,1,296)=83.2609;
		MAT_ELEM(angles,0,297)=310.1881;MAT_ELEM(angles,1,297)=83.2609;
		MAT_ELEM(angles,0,298)=62.701;	MAT_ELEM(angles,1,298)=58.6205;
		MAT_ELEM(angles,0,299)=22.1881;	MAT_ELEM(angles,1,299)=83.2609;
		MAT_ELEM(angles,0,300)=49.8119;	MAT_ELEM(angles,1,300)=83.2609;
		MAT_ELEM(angles,0,301)=206.701;	MAT_ELEM(angles,1,301)=58.621;
		MAT_ELEM(angles,0,302)=153.299;	MAT_ELEM(angles,1,302)=58.621;
		MAT_ELEM(angles,0,303)=180;		MAT_ELEM(angles,1,303)=39.647;
		MAT_ELEM(angles,0,304)=252;		MAT_ELEM(angles,1,304)=39.647;
		MAT_ELEM(angles,0,305)=108;		MAT_ELEM(angles,1,305)=39.647;
		MAT_ELEM(angles,0,306)=324;		MAT_ELEM(angles,1,306)=55.5056;
		MAT_ELEM(angles,0,307)=36;		MAT_ELEM(angles,1,307)=55.5056;
		MAT_ELEM(angles,0,308)=27.3931;	MAT_ELEM(angles,1,308)=61.2449;
		MAT_ELEM(angles,0,309)=175.0477;MAT_ELEM(angles,1,309)=69.933;
		MAT_ELEM(angles,0,310)=112.9523;MAT_ELEM(angles,1,310)=69.933;
		MAT_ELEM(angles,0,311)=184.9523;MAT_ELEM(angles,1,311)=69.933;
		MAT_ELEM(angles,0,312)=247.0477;MAT_ELEM(angles,1,312)=69.933;
		MAT_ELEM(angles,0,313)=256.9523;MAT_ELEM(angles,1,313)=69.933;
		MAT_ELEM(angles,0,314)=103.0477;MAT_ELEM(angles,1,314)=69.933;
		MAT_ELEM(angles,0,315)=243.3931;MAT_ELEM(angles,1,315)=61.245;
		MAT_ELEM(angles,0,316)=116.6069;MAT_ELEM(angles,1,316)=61.245;
		MAT_ELEM(angles,0,317)=180;		MAT_ELEM(angles,1,317)=7.929;
		MAT_ELEM(angles,0,318)=252;		MAT_ELEM(angles,1,318)=7.929;
		MAT_ELEM(angles,0,319)=99.3931;	MAT_ELEM(angles,1,319)=61.245;
		MAT_ELEM(angles,0,320)=108;		MAT_ELEM(angles,1,320)=7.929;
	}
	else
	{
		angles.initZeros(2,81);
		MAT_ELEM(angles, 0, 0) = 0.000000;	 	 MAT_ELEM(angles, 1, 0) = 0.000000;
		MAT_ELEM(angles, 0, 1) = 36.000000;	 	 MAT_ELEM(angles, 1, 1) = 15.858741;
		MAT_ELEM(angles, 0, 2) = 36.000000;	 	 MAT_ELEM(angles, 1, 2) = 31.717482;
		MAT_ELEM(angles, 0, 3) = 36.000000;	 	 MAT_ELEM(angles, 1, 3) = 47.576224;
		MAT_ELEM(angles, 0, 4) = 36.000000;	 	 MAT_ELEM(angles, 1, 4) = 63.434965;
		MAT_ELEM(angles, 0, 5) = 62.494295;	 	 MAT_ELEM(angles, 1, 5) = -76.558393;
		MAT_ELEM(angles, 0, 6) = 54.000000;	 	 MAT_ELEM(angles, 1, 6) = 90.000000;
		MAT_ELEM(angles, 0, 7) = 45.505705;	 	 MAT_ELEM(angles, 1, 7) = 76.558393;
		MAT_ELEM(angles, 0, 8) = 108.000000;	 MAT_ELEM(angles, 1, 8) = 15.858741;
		MAT_ELEM(angles, 0, 9) = 108.000000;	 MAT_ELEM(angles, 1, 9) = 31.717482;
		MAT_ELEM(angles, 0, 10) = 108.000000;	 MAT_ELEM(angles, 1, 10) = 47.576224;
		MAT_ELEM(angles, 0, 11) = 108.000000;	 MAT_ELEM(angles, 1, 11) = 63.434965;
		MAT_ELEM(angles, 0, 12) = 134.494295;	 MAT_ELEM(angles, 1, 12) = -76.558393;
		MAT_ELEM(angles, 0, 13) = 126.000000;	 MAT_ELEM(angles, 1, 13) = 90.000000;
		MAT_ELEM(angles, 0, 14) = 117.505705;	 MAT_ELEM(angles, 1, 14) = 76.558393;
		MAT_ELEM(angles, 0, 15) = 144.000000;	 MAT_ELEM(angles, 1, 15) = -15.858741;
		MAT_ELEM(angles, 0, 16) = 144.000000;	 MAT_ELEM(angles, 1, 16) = -31.717482;
		MAT_ELEM(angles, 0, 17) = 144.000000;	 MAT_ELEM(angles, 1, 17) = -47.576224;
		MAT_ELEM(angles, 0, 18) = 144.000000;	 MAT_ELEM(angles, 1, 18) = -63.434965;
		MAT_ELEM(angles, 0, 19) = 170.494295;	 MAT_ELEM(angles, 1, 19) = 76.558393;
		MAT_ELEM(angles, 0, 20) = 162.000000;	 MAT_ELEM(angles, 1, 20) = 90.000000;
		MAT_ELEM(angles, 0, 21) = 153.505705;	 MAT_ELEM(angles, 1, 21) = -76.558393;
		MAT_ELEM(angles, 0, 22) = 72.000000;	 MAT_ELEM(angles, 1, 22) = -15.858741;
		MAT_ELEM(angles, 0, 23) = 72.000000;	 MAT_ELEM(angles, 1, 23) = -31.717482;
		MAT_ELEM(angles, 0, 24) = 72.000000;	 MAT_ELEM(angles, 1, 24) = -47.576224;
		MAT_ELEM(angles, 0, 25) = 72.000000;	 MAT_ELEM(angles, 1, 25) = -63.434965;
		MAT_ELEM(angles, 0, 26) = 98.494295;	 MAT_ELEM(angles, 1, 26) = 76.558393;
		MAT_ELEM(angles, 0, 27) = 90.000000;	 MAT_ELEM(angles, 1, 27) = 90.000000;
		MAT_ELEM(angles, 0, 28) = 81.505705;	 MAT_ELEM(angles, 1, 28) = -76.558393;
		MAT_ELEM(angles, 0, 29) = 0.000000;	 	 MAT_ELEM(angles, 1, 29) = -15.858741;
		MAT_ELEM(angles, 0, 30) = 0.000000;	 	 MAT_ELEM(angles, 1, 30) = -31.717482;
		MAT_ELEM(angles, 0, 31) = 0.000000;	 	 MAT_ELEM(angles, 1, 31) = -47.576224;
		MAT_ELEM(angles, 0, 32) = 0.000000;	 	 MAT_ELEM(angles, 1, 32) = -63.434965;
		MAT_ELEM(angles, 0, 33) = 26.494295;	 MAT_ELEM(angles, 1, 33) = 76.558393;
		MAT_ELEM(angles, 0, 34) = 18.000000;	 MAT_ELEM(angles, 1, 34) = 90.000000;
		MAT_ELEM(angles, 0, 35) = 9.505705;	 	 MAT_ELEM(angles, 1, 35) = -76.558393;
		MAT_ELEM(angles, 0, 36) = 12.811021;	 MAT_ELEM(angles, 1, 36) = 42.234673;
		MAT_ELEM(angles, 0, 37) = 18.466996;	 MAT_ELEM(angles, 1, 37) = 59.620797;
		MAT_ELEM(angles, 0, 38) = 0.000000;	 	 MAT_ELEM(angles, 1, 38) = 90.000000;
		MAT_ELEM(angles, 0, 39) = 8.867209;	 	 MAT_ELEM(angles, 1, 39) = 75.219088;
		MAT_ELEM(angles, 0, 40) = 72.000000;	 MAT_ELEM(angles, 1, 40) = 26.565058;
		MAT_ELEM(angles, 0, 41) = 59.188979;	 MAT_ELEM(angles, 1, 41) = 42.234673;
		MAT_ELEM(angles, 0, 42) = 84.811021;	 MAT_ELEM(angles, 1, 42) = 42.234673;
		MAT_ELEM(angles, 0, 43) = 53.533003;	 MAT_ELEM(angles, 1, 43) = 59.620797;
		MAT_ELEM(angles, 0, 44) = 72.000000;	 MAT_ELEM(angles, 1, 44) = 58.282544;
		MAT_ELEM(angles, 0, 45) = 90.466996;	 MAT_ELEM(angles, 1, 45) = 59.620797;
		MAT_ELEM(angles, 0, 46) = 72.000000;	 MAT_ELEM(angles, 1, 46) = 90.000000;
		MAT_ELEM(angles, 0, 47) = 63.132791;	 MAT_ELEM(angles, 1, 47) = 75.219088;
		MAT_ELEM(angles, 0, 48) = 80.867209;	 MAT_ELEM(angles, 1, 48) = 75.219088;
		MAT_ELEM(angles, 0, 49) = 144.000000;	 MAT_ELEM(angles, 1, 49) = 26.565058;
		MAT_ELEM(angles, 0, 50) = 131.188979;	 MAT_ELEM(angles, 1, 50) = 42.234673;
		MAT_ELEM(angles, 0, 51) = 156.811021;	 MAT_ELEM(angles, 1, 51) = 42.234673;
		MAT_ELEM(angles, 0, 52) = 125.533003;	 MAT_ELEM(angles, 1, 52) = 59.620797;
		MAT_ELEM(angles, 0, 53) = 144.000000;	 MAT_ELEM(angles, 1, 53) = 58.282544;
		MAT_ELEM(angles, 0, 54) = 162.466996;	 MAT_ELEM(angles, 1, 54) = 59.620797;
		MAT_ELEM(angles, 0, 55) = 144.000000;	 MAT_ELEM(angles, 1, 55) = 90.000000;
		MAT_ELEM(angles, 0, 56) = 135.132791;	 MAT_ELEM(angles, 1, 56) = 75.219088;
		MAT_ELEM(angles, 0, 57) = 152.867209;	 MAT_ELEM(angles, 1, 57) = 75.219088;
		MAT_ELEM(angles, 0, 58) = 180.000000;	 MAT_ELEM(angles, 1, 58) = -26.565058;
		MAT_ELEM(angles, 0, 59) = 167.188979;	 MAT_ELEM(angles, 1, 59) = -42.234673;
		MAT_ELEM(angles, 0, 60) = 180.000000;	 MAT_ELEM(angles, 1, 60) = -58.282544;
		MAT_ELEM(angles, 0, 61) = 161.533003;	 MAT_ELEM(angles, 1, 61) = -59.620797;
		MAT_ELEM(angles, 0, 62) = 171.132791;	 MAT_ELEM(angles, 1, 62) = -75.219088;
		MAT_ELEM(angles, 0, 63) = 108.000000;	 MAT_ELEM(angles, 1, 63) = -26.565058;
		MAT_ELEM(angles, 0, 64) = 120.811021;	 MAT_ELEM(angles, 1, 64) = -42.234673;
		MAT_ELEM(angles, 0, 65) = 95.188979;	 MAT_ELEM(angles, 1, 65) = -42.234673;
		MAT_ELEM(angles, 0, 66) = 126.466996;	 MAT_ELEM(angles, 1, 66) = -59.620797;
		MAT_ELEM(angles, 0, 67) = 108.000000;	 MAT_ELEM(angles, 1, 67) = -58.282544;
		MAT_ELEM(angles, 0, 68) = 89.533003;	 MAT_ELEM(angles, 1, 68) = -59.620797;
		MAT_ELEM(angles, 0, 69) = 108.000000;	 MAT_ELEM(angles, 1, 69) = 90.000000;
		MAT_ELEM(angles, 0, 70) = 116.867209;	 MAT_ELEM(angles, 1, 70) = -75.219088;
		MAT_ELEM(angles, 0, 71) = 99.132791;	 MAT_ELEM(angles, 1, 71) = -75.219088;
		MAT_ELEM(angles, 0, 72) = 36.000000;	 MAT_ELEM(angles, 1, 72) = -26.565058;
		MAT_ELEM(angles, 0, 73) = 48.811021;	 MAT_ELEM(angles, 1, 73) = -42.234673;
		MAT_ELEM(angles, 0, 74) = 23.188979;	 MAT_ELEM(angles, 1, 74) = -42.234673;
		MAT_ELEM(angles, 0, 75) = 54.466996;	 MAT_ELEM(angles, 1, 75) = -59.620797;
		MAT_ELEM(angles, 0, 76) = 36.000000;	 MAT_ELEM(angles, 1, 76) = -58.282544;
		MAT_ELEM(angles, 0, 77) = 17.533003;	 MAT_ELEM(angles, 1, 77) = -59.620797;
		MAT_ELEM(angles, 0, 78) = 36.000000;	 MAT_ELEM(angles, 1, 78) = 90.000000;
		MAT_ELEM(angles, 0, 79) = 44.867209;	 MAT_ELEM(angles, 1, 79) = -75.219088;
		MAT_ELEM(angles, 0, 80) = 27.132791;	 MAT_ELEM(angles, 1, 80) = -75.219088;
	}

	angles *= PI/180;
}


void ProgFSO::anistropyParameter(const MultidimArray<double> &FSC,
		MultidimArray<double> &directionAnisotropy, size_t dirnumber,
		MultidimArray<double> &	aniParam, double thrs)
{
	double N = 0;
	for (size_t k = 0; k<aniParam.nzyxdim; k++)
	{
		if (DIRECT_MULTIDIM_ELEM(FSC, k) >= thrs)
		{
			DIRECT_MULTIDIM_ELEM(aniParam, k) += 1.0;
			N++;
		}
	}
	DIRECT_MULTIDIM_ELEM(directionAnisotropy, dirnumber) = N;
}


void ProgFSO::prepareData(MultidimArray<double> &half1, MultidimArray<double> &half2)
{


	
	std::cout << "Reading data..." << std::endl;
	Image<double> imgHalf1, imgHalf2;
	imgHalf1.read(fnhalf1);
	imgHalf2.read(fnhalf2);

	half1 = imgHalf1();
	half2 = imgHalf2();

	// Applying the mask
	if (fnmask!="")
	{
		Image<double> mask;
		MultidimArray<double> &pmask = mask();
		mask.read(fnmask);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(pmask)
		{
			double valmask = (double) DIRECT_MULTIDIM_ELEM(pmask, n);
			DIRECT_MULTIDIM_ELEM(half1, n) = DIRECT_MULTIDIM_ELEM(half1, n) * valmask;
			DIRECT_MULTIDIM_ELEM(half2, n) = DIRECT_MULTIDIM_ELEM(half2, n) * valmask;
		}
	}

	half1.setXmippOrigin();
	half2.setXmippOrigin();

	// fftshift must by applied before computing the fft. This will be computed later
	CenterFFT(half1, true);
	CenterFFT(half2, true);
}


void ProgFSO::saveAnisotropyToMetadata(MetaData &mdAnisotropy,
		const MultidimArray<double> &freq,
		const MultidimArray<double> &anisotropy)
{
	size_t objId;
	FOR_ALL_ELEMENTS_IN_ARRAY1D(anisotropy)
	{
		if (i>0)
		{
		objId = mdAnisotropy.addObject();
		mdAnisotropy.setValue(MDL_RESOLUTION_FREQ, dAi(freq, i),objId);
		mdAnisotropy.setValue(MDL_RESOLUTION_FSO, dAi(anisotropy, i),objId);
		mdAnisotropy.setValue(MDL_RESOLUTION_FREQREAL, 1.0/dAi(freq, i),objId);
		}
	}
	mdAnisotropy.write(fnOut+"/fso.xmd");
}


void ProgFSO::directionalFilter(MultidimArray<std::complex<double>> &FThalf1, 
			MultidimArray<double> &threeDfsc, 
			MultidimArray<double> &filteredMap, int m1sizeX, int m1sizeY, int m1sizeZ)
    {
    	Image<double> imgHalf1, imgHalf2;
    	imgHalf1.read(fnhalf1);
		imgHalf2.read(fnhalf2);

    	auto &half1 = imgHalf1();
    	auto &half2 = imgHalf2();

        FourierTransformer transformer1(FFTW_BACKWARD);
        transformer1.FourierTransform(half1, FThalf1);//, false);
		FourierTransformer transformer2(FFTW_BACKWARD);
		MultidimArray<std::complex<double>> FThalf2;
		FThalf2.resizeNoCopy(FThalf1);
        transformer1.FourierTransform(half2, FThalf2, false);

    	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(threeDfsc)
        {
    		DIRECT_MULTIDIM_ELEM(FThalf1, n) += DIRECT_MULTIDIM_ELEM(FThalf2, n);
			DIRECT_MULTIDIM_ELEM(FThalf1, n) *= DIRECT_MULTIDIM_ELEM(threeDfsc, n); 
    	}
    	filteredMap.resizeNoCopy(m1sizeX, m1sizeY, m1sizeZ);
    	transformer1.inverseFourierTransform(FThalf1, filteredMap);
    }


void ProgFSO::resolutionDistribution(MultidimArray<double> &resDirFSC, FileName &fn)
    {
    	Matrix2D<int> anglesResolution;
    	const size_t Nrot = 360;
    	const size_t Ntilt = 91;
    	size_t objIdOut;

    	MetaData mdOut;
    	Matrix2D<double> w, wt;
    	w.initZeros(Nrot, Ntilt);
    	wt = w;
    	float cosAngle = cosf(ang_con);
    	float aux = 4.0/((cosAngle -1)*(cosAngle -1));
    	// Directional resolution is store in a metadata
			
		for (int i=0; i<Nrot; i++)
		{
			float rotmatrix =  i*PI/180.0;

			for (int j=0; j<Ntilt; j++)
			{
				float tiltmatrix = j*PI/180.0;
				// position on the spehere
				float xx = sinf(tiltmatrix)*cosf(rotmatrix);
				float yy = sinf(tiltmatrix)*sinf(rotmatrix);
				float zz = cosf(tiltmatrix);

				// initializing the weights
				double w = 0;
				double wt = 0;

				for (size_t k = 0; k<angles.mdimx; k++)
				{

					float rot = MAT_ELEM(angles, 0, k);
					float tilt = MAT_ELEM(angles, 1, k);

					// position of the direction on the sphere
					float x_dir = sinf(tilt)*cosf(rot);
					float y_dir = sinf(tilt)*sinf(rot);
					float z_dir = cosf(tilt);


					float cosine = fabs(x_dir*xx + y_dir*yy + z_dir*zz);
					if (cosine>=cosAngle)
					{
						cosine = expf( -((cosine -1)*(cosine -1))*aux );
						w += cosine*( dAi(resDirFSC, k) );
						wt += cosine;
					}
				}

			double wRes = w/wt;
			objIdOut = mdOut.addObject();
			mdOut.setValue(MDL_ANGLE_ROT, (double) i, objIdOut);
			mdOut.setValue(MDL_ANGLE_TILT, (double) j, objIdOut);
			mdOut.setValue(MDL_RESOLUTION_FRC, wRes, objIdOut);
			}
		}

		mdOut.write(fn);
    }



void ProgFSO::getCompleteFourier(MultidimArray<double> &V, MultidimArray<double> &newV,
    		int m1sizeX, int m1sizeY, int m1sizeZ)
        {
    	newV.resizeNoCopy(m1sizeX, m1sizeY, m1sizeZ);
		int ndim=3;
		if (m1sizeX==1)
		{
			ndim=2;
			if (m1sizeY==1)
				ndim=1;
		}
		double *ptrSource=NULL;
		double *ptrDest=NULL;
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(newV)
		{
			ptrDest=(double*)&DIRECT_A3D_ELEM(newV,k,i,j);
			if (j<XSIZE(V))
			{
				ptrSource=(double*)&DIRECT_A3D_ELEM(V,k,i,j);
				*ptrDest=*ptrSource;
			}
			else
			{
				ptrSource=(double*)&DIRECT_A3D_ELEM(V,
													(m1sizeZ-k)%m1sizeZ,
													(m1sizeY-i)%m1sizeY,
													m1sizeX-j);
				*ptrDest=*ptrSource;
			}
		}
   }


void ProgFSO::createFullFourier(MultidimArray<double> &fourierHalf, FileName &fnMap,
    		int m1sizeX, int m1sizeY, int m1sizeZ)
    {
    	MultidimArray<double> fullMap;
		getCompleteFourier(fourierHalf, fullMap, m1sizeX, m1sizeY, m1sizeZ);
		CenterFFT(fullMap, true);
		Image<double> saveImg;
		saveImg() = fullMap;
	    saveImg.write(fnMap);
    }


void ProgFSO::run()
	{
		std::cout << "Starting ... " << std::endl << std::endl;
		
		MultidimArray<double> half1, half2;
    	MultidimArray<double> &phalf1 = half1, &phalf2 = half2;

		//This read the data and applies a fftshift to in the next step compute the fft
    	prepareData(half1, half2);

		//Computing the FFT
		FourierTransformer transformer2(FFTW_BACKWARD), transformer1(FFTW_BACKWARD);
		transformer1.setThreadsNumber(Nthreads);
		transformer2.setThreadsNumber(Nthreads);

        transformer1.FourierTransform(half1, FT1, false);
     	transformer2.FourierTransform(half2, FT2, false);

		// Defining frequencies freq_fourier_x,y,z and freqMap
		// The number of frequencies in each shell freqElem is determined
        defineFrequencies(FT1, phalf1);

		half1.clear();
		half2.clear();

		// Storing the shell of both maps as vectors global
		// The global FSC is also computed
		MultidimArray<double> freq;
		arrangeFSC_and_fscGlobal(sampling, thrs, freq);

		std::cout << " " << std::endl;
		FT2.clear();

		// Generating the set of directions to be analyzed
		// And converting the cone angle to radians
    	generateDirections(angles, true);
    	ang_con = ang_con*PI/180;

		// Preparing the metadata for storing the FSO
		MultidimArray<double> directionAnisotropy(angles.mdimx), resDirFSC(angles.mdimx), aniParam;
    	aniParam.initZeros(xvoldim/2+1);

		// Computing directional FSC and 3DFSC
		MultidimArray<double> fsc, threeD_FSC, normalizationMap;
		double resInterp = -1;
		threeD_FSC.resizeNoCopy(real_z1z2);
		threeD_FSC.initZeros();
		normalizationMap.resizeNoCopy(real_z1z2);
		normalizationMap.initZeros();
		
    	for (size_t k = 0; k<angles.mdimx; k++)
		{
			float rot  = MAT_ELEM(angles, 0, k);
			float tilt = MAT_ELEM(angles, 1, k);

			// Estimating the direction FSC along the direction given by rot and tilt
			fscDir_fast(fsc, rot, tilt, threeD_FSC, normalizationMap, thrs, resInterp, k);

			printf ("Direction %zu/%zu -> %.2f A \n", k, angles.mdimx, resInterp);

			dAi(resDirFSC, k) = resInterp;
			
			// Updating the FSO curve
			anistropyParameter(fsc, directionAnisotropy, k, aniParam, thrs);
		}

		std::cout << "----- Directional resolution estimated -----" <<  std::endl <<  std::endl;
    	std::cout << "Preparing results ..." <<  std::endl;

    	// ANISOTROPY CURVE
    	aniParam /= (double) angles.mdimx;
    	MetaData mdani;
		saveAnisotropyToMetadata(mdani, freq, aniParam);
		FileName fn;
		
		if (do_3dfsc_filter)
		{
			// HALF 3DFSC MAP
			MultidimArray<double> d3_FSCMap;
			MultidimArray<double> d3_aniFilter;
			d3_FSCMap.resizeNoCopy(FT1);
			d3_FSCMap.initConstant(0);
			d3_aniFilter.resizeNoCopy(FT1);
			d3_aniFilter.initConstant(0);

			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(threeD_FSC)
			{
					double value = DIRECT_MULTIDIM_ELEM(threeD_FSC, n) /= DIRECT_MULTIDIM_ELEM(normalizationMap, n);
					if (std::isnan(value) == 1)
						value = 1.0;
					
					if ((DIRECT_MULTIDIM_ELEM(threeD_FSC, n)> thrs) )//&& (DIRECT_MULTIDIM_ELEM(aniFilter, n) <1))
						DIRECT_MULTIDIM_ELEM(aniFilter, n) = 1;
					
					size_t idx = DIRECT_MULTIDIM_ELEM(arr2indx, n);
					DIRECT_MULTIDIM_ELEM(d3_FSCMap, idx) = value;
					DIRECT_MULTIDIM_ELEM(d3_aniFilter, idx) = DIRECT_MULTIDIM_ELEM(aniFilter, n);
			}

			// This code fix the empty line line in Fourier space
			size_t auxVal;
			auxVal = YSIZE(d3_FSCMap)/2;
			
			size_t j = 0;
			for(size_t i=0; i<YSIZE(d3_FSCMap); ++i)
			{
				if (i>auxVal)
				{
					for(size_t k=0; k<ZSIZE(d3_FSCMap); ++k)
					{
						DIRECT_A3D_ELEM(d3_FSCMap,k,i,j) = DIRECT_A3D_ELEM(d3_FSCMap,k,i,j+1);
						DIRECT_A3D_ELEM(d3_aniFilter,k,i,j) = DIRECT_A3D_ELEM(d3_aniFilter,k,i,j+1);
					}
				}
			}

			double sigma = 3;

			realGaussianFilter(d3_aniFilter, sigma);

			// DIRECTIONAL FILTERED MAP
			MultidimArray<double> filteredMap;
			directionalFilter(FT1, d3_aniFilter, filteredMap, xvoldim, yvoldim, zvoldim);
			Image<double> saveImg2;
			saveImg2() = filteredMap;
			saveImg2.write(fnOut+"/filteredMap.mrc");
			
			
			//FULL 3DFSC MAP

			fn = fnOut+"/3dFSC.mrc";
			createFullFourier(d3_FSCMap, fn, xvoldim, yvoldim, zvoldim);
			
		}

		// DIRECTIONAL RESOLUTION DISTRIBUTION ON THE PROJECTION SPHERE
		fn = fnOut+"/Resolution_Distribution.xmd";
		
		resolutionDistribution(resDirFSC, fn);
		
		std::cout << "-------------Finished-------------" << std::endl;
}
