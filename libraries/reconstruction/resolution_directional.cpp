/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 * 			   Carlos Oscar S. Sorzano            coss@cnb.csic.es (2016)
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

#include "resolution_directional.h"
//#define DEBUG
//#define DEBUG_MASK
//#define DEBUG_DIR
//#define DEBUG_FILTER
//#define MONO_AMPLITUDE
//define DEBUG_SYMMETRY

void ProgResDir::readParams()
{
	fnVol = getParam("--vol");
	fnOut = getParam("-o");
	fnMask = getParam("--mask");
	sampling = getDoubleParam("--sampling_rate");
	R = getDoubleParam("--volumeRadius");
	significance = getDoubleParam("--significance");
	res_step = getDoubleParam("--resStep");
	fnradial = getParam("--radialRes");
	fnazimuthal = getParam("--azimuthalRes");
	fnRadialAvG = getParam("--radialAvG");
	fnHighestResolution = getParam("--highestResolutionVol");
	fnLowestResolution = getParam("--lowestResolutionVol");
	fnDoa1 = getParam("--doa1");
	fnDoa2 = getParam("--doa2");
	fnMDThr = getParam("--radialAzimuthalThresholds");
	fnMonoRes = getParam("--monores");
	fnprefMin = getParam("--prefMin");
	fnZscore = getParam("--zScoremap");
	Nthr = getIntParam("--threads");
	fastCompute = checkParam("--fast");
}


void ProgResDir::defineParams()
{
	addUsageLine("This function determines the local resolution of a map");
	addParamsLine("  --vol <vol_file=\"\">                   : Input volume");
	addParamsLine("  --mask <vol_file=\"\">                  : Mask defining the macromolecule");
	addParamsLine("  -o <output=\"MGresolution.vol\">        : Local resolution volume (in Angstroms)");
	addParamsLine("  [--sampling_rate <s=1>]                 : Sampling rate (A/px)");
	addParamsLine("  [--resStep <s=0.5>]  		             : Resolution step (precision) in A");
	addParamsLine("  [--volumeRadius <s=100>]                : This parameter determines the radius of a sphere where the volume is");
	addParamsLine("  [--significance <s=0.95>]               : The level of confidence for the hypothesis test.");
	addParamsLine("  --radialRes <vol_file=\"\">             : Output radial resolution map");
	addParamsLine("  --azimuthalRes <vol_file=\"\">          : Output azimuthal resolution map");
	addParamsLine("  --highestResolutionVol <vol_file=\"\">  : Output highest resolution map");
	addParamsLine("  --lowestResolutionVol <vol_file=\"\">  : Output lowest resolution map");
	addParamsLine("  --doa1 <vol_file=\"\">                  : Output doa1 map");
	addParamsLine("  --doa2 <vol_file=\"\">                  : Output doa2 map");
	addParamsLine("  --radialAzimuthalThresholds <vol_file=\"\">  : Radial and azimuthal threshold for representation resolution maps");
	addParamsLine("  --radialAvG <md_file=\"\">              : Radial Average of higesht, lowest, azimuthal, and radial, and local resolution map");
	addParamsLine("  --monores <vol_file=\"\">             : Local resolution map");
	addParamsLine("  --prefMin <vol_file=\"\">               : Metadata of highest resolution per direction");
	addParamsLine("  [--threads <s=4>]                       : Number of threads");
	addParamsLine("  --zScoremap <vol_file=\"\">             : Local zScore map, voxel with zscore higher than 3 are weird");
	addParamsLine("  [--fast]                                : Fast computation");
}

void ProgResDir::produceSideInfo()
{

	std::cout << "Starting..." << std::endl;

	Image<double> V;
	V.read(fnVol);

	V().setXmippOrigin();

	//Sweeping the projection sphere
	std::cout << "Obtaining angular projections..." << std::endl;
	generateGridProjectionMatching(angles);

	FourierTransformer transformer;

	MultidimArray<double> &inputVol = V();
	VRiesz.resizeNoCopy(inputVol);
	N_freq = ZSIZE(inputVol);
	maxRes = ZSIZE(inputVol);
	minRes = 2*sampling;

	transformer_inv.setThreadsNumber(Nthr);

	transformer.FourierTransform(inputVol, fftV);
	iu.initZeros(fftV);

	// Frequency volume
	double uz, uy, ux, uz2, u2, uz2y2;
	long n=0;

	for(size_t k=0; k<ZSIZE(fftV); ++k)
	{
		FFT_IDX2DIGFREQ(k,ZSIZE(inputVol),uz);
		uz2=uz*uz;

		for(size_t i=0; i<YSIZE(fftV); ++i)
		{
			FFT_IDX2DIGFREQ(i,YSIZE(inputVol),uy);
			uz2y2=uz2+uy*uy;

			for(size_t j=0; j<XSIZE(fftV); ++j)
			{
				FFT_IDX2DIGFREQ(j,XSIZE(inputVol), ux);
				u2=uz2y2+ux*ux;
				if ((k != 0) || (i != 0) || (j != 0))
					DIRECT_MULTIDIM_ELEM(iu,n) = 1.0/sqrt(u2);
				else
					DIRECT_MULTIDIM_ELEM(iu,n) = 1e38;
				++n;
			}
		}
	}

	// Prepare mask
	MultidimArray<int> &pMask=mask();

	if (fnMask != "")
	{
		mask.read(fnMask);
		mask().setXmippOrigin();
	}
	else
	{
		std::cout << "Error: a mask ought to be provided" << std::endl;
		exit(0);
	}

	N_smoothing = 7;
	NVoxelsOriginalMask = 0;
	double radius = 0;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(pMask)
	{
		if (A3D_ELEM(pMask, k, i, j) == 1)
		{
			if ((k*k + i*i + j*j)>radius)
				radius = k*k + i*i + j*j;
		}
//		std::cout << "i j k " << i << " " << j << " " << k << std::endl;

		if (A3D_ELEM(pMask, k, i, j) == 1)
			++NVoxelsOriginalMask;
		if (i*i+j*j+k*k > (R-N_smoothing)*(R-N_smoothing))
			A3D_ELEM(pMask, k, i, j) = -1;
	}
	Rparticle = round(sqrt(radius));
	std::cout << "particle radius = " << Rparticle << std::endl;
	size_t xrows = angles.mdimx;

	resolutionMatrix.initConstant(xrows, NVoxelsOriginalMask, maxRes);


	std::cout << MAT_ELEM(resolutionMatrix, 1, 2) << std::endl;



	#ifdef DEBUG_MASK
	std::cout << "-------------DEBUG-----------" <<std::endl;
	std::cout << "Next number ought to be the same than number of directions"
			<< std::endl;
	std::cout << "number of angles" << xrows << std::endl;
	std::cout << "---------END-DEBUG--------" <<std::endl;
	mask.write("mask.vol");
	#endif

	freq_fourier.initZeros(ZSIZE(inputVol));
	int size = ZSIZE(inputVol);
	maxRes = size;
	minRes = 1;
	V.clear();


	double u;
	int size_fourier(ZSIZE(fftV));

	VEC_ELEM(freq_fourier,0) = 1e-38;
	for(size_t k=1; k<size_fourier; ++k)
	{
		FFT_IDX2DIGFREQ(k,size, u);
		VEC_ELEM(freq_fourier,k) = u;
	}
}


void ProgResDir::generateGridProjectionMatching(Matrix2D<double> &angles)
{
  	if (fastCompute == false)
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
	else
	{
		angles.initZeros(2,47);
		MAT_ELEM(angles, 0,1) =0;    		MAT_ELEM(angles, 1,1) =0;
		MAT_ELEM(angles, 0,2) =36;    		MAT_ELEM(angles, 1,2) =21.145;
		MAT_ELEM(angles, 0,3) =36;    		MAT_ELEM(angles, 1,3) =42.29;
		MAT_ELEM(angles, 0,4) =36;    		MAT_ELEM(angles, 1,4) =63.435;
		MAT_ELEM(angles, 0,5) =59.6043;    	MAT_ELEM(angles, 1,5) =-81.0207;
		MAT_ELEM(angles, 0,6) =48.3957;    	MAT_ELEM(angles, 1,6) =81.0207;
		MAT_ELEM(angles, 0,7) =108;    		MAT_ELEM(angles, 1,7) =21.145;
		MAT_ELEM(angles, 0,8) =108;    		MAT_ELEM(angles, 1,8) =42.29;
		MAT_ELEM(angles, 0,9) =108;    		MAT_ELEM(angles, 1,9) =63.435;
		MAT_ELEM(angles, 0,10) =131.6043;   MAT_ELEM(angles, 1,10) =-81.0207;
		MAT_ELEM(angles, 0,11) =120.3957;   MAT_ELEM(angles, 1,11) =81.0207;
		MAT_ELEM(angles, 0,12) =144;    	MAT_ELEM(angles, 1,12) =-21.145;
		MAT_ELEM(angles, 0,13) =144;    	MAT_ELEM(angles, 1,13) =-42.29;
		MAT_ELEM(angles, 0,14) =144;    	MAT_ELEM(angles, 1,14) =-63.435;
		MAT_ELEM(angles, 0,15) =167.6043;   MAT_ELEM(angles, 1,15) =81.0207;
		MAT_ELEM(angles, 0,16) =156.3957;   MAT_ELEM(angles, 1,16) =-81.0207;
		MAT_ELEM(angles, 0,17) =72;    		MAT_ELEM(angles, 1,17) =-21.145;
		MAT_ELEM(angles, 0,18) =72;    		MAT_ELEM(angles, 1,18) =-42.29;
		MAT_ELEM(angles, 0,19) =72;    		MAT_ELEM(angles, 1,19) =-63.435;
		MAT_ELEM(angles, 0,20) =95.6043;    MAT_ELEM(angles, 1,20) =81.0207;
		MAT_ELEM(angles, 0,21) =84.3957;    MAT_ELEM(angles, 1,21) =-81.0207;
		MAT_ELEM(angles, 0,22) =0;    		MAT_ELEM(angles, 1,22) =-21.145;
		MAT_ELEM(angles, 0,23) =0;    		MAT_ELEM(angles, 1,23) =-42.29;
		MAT_ELEM(angles, 0,24) =0;    		MAT_ELEM(angles, 1,24) =-63.435;
		MAT_ELEM(angles, 0,25) =23.6043;    MAT_ELEM(angles, 1,25) =81.0207;
		MAT_ELEM(angles, 0,26) =12.3957;    MAT_ELEM(angles, 1,26) =-81.0207;
		MAT_ELEM(angles, 0,27) =12.3756;    MAT_ELEM(angles, 1,27) =58.8818;
		MAT_ELEM(angles, 0,28) =72;    		MAT_ELEM(angles, 1,28) =36.349;
		MAT_ELEM(angles, 0,29) =59.6244;    MAT_ELEM(angles, 1,29) =58.8818;
		MAT_ELEM(angles, 0,30) =84.3756;    MAT_ELEM(angles, 1,30) =58.8818;
		MAT_ELEM(angles, 0,31) =72;    		MAT_ELEM(angles, 1,31) =80.2161;
		MAT_ELEM(angles, 0,32) =144;    	MAT_ELEM(angles, 1,32) =36.349;
		MAT_ELEM(angles, 0,33) =131.6244;   MAT_ELEM(angles, 1,33) =58.8818;
		MAT_ELEM(angles, 0,34) =156.3756;   MAT_ELEM(angles, 1,34) =58.8818;
		MAT_ELEM(angles, 0,35) =144;    	MAT_ELEM(angles, 1,35) =80.2161;
		MAT_ELEM(angles, 0,36) =180;    	MAT_ELEM(angles, 1,36) =-36.349;
		MAT_ELEM(angles, 0,37) =167.6244;   MAT_ELEM(angles, 1,37) =-58.8818;
		MAT_ELEM(angles, 0,38) =180;    	MAT_ELEM(angles, 1,38) =-80.2161;
		MAT_ELEM(angles, 0,39) =108;    	MAT_ELEM(angles, 1,39) =-36.349;
		MAT_ELEM(angles, 0,40) =120.3756;   MAT_ELEM(angles, 1,40) =-58.8818;
		MAT_ELEM(angles, 0,41) =95.6244;    MAT_ELEM(angles, 1,41) =-58.8818;
		MAT_ELEM(angles, 0,42) =108;    	MAT_ELEM(angles, 1,42) =-80.2161;
		MAT_ELEM(angles, 0,43) =36;    		MAT_ELEM(angles, 1,43) =-36.349;
		MAT_ELEM(angles, 0,44) =48.3756;    MAT_ELEM(angles, 1,44) =-58.8818;
		MAT_ELEM(angles, 0,45) =23.6244;    MAT_ELEM(angles, 1,45) =-58.8818;
		MAT_ELEM(angles, 0,46) =36;    		MAT_ELEM(angles, 1,46) =-80.2161;
	}

}



void ProgResDir::amplitudeMonogenicSignal3D_fast(const MultidimArray< std::complex<double> > &myfftV,
		double freq, double freqH, double freqL, MultidimArray<double> &amplitude, int count, int dir, FileName fnDebug,
		double rot, double tilt)
{
	fftVRiesz.initZeros(myfftV);
//	MultidimArray<double> coneVol;
//	coneVol.initZeros(myfftV);
	fftVRiesz_aux.initZeros(myfftV);
	std::complex<double> J(0,1);

	// Filter the input volume and add it to amplitude
	long n=0;
	double ideltal=PI/(freq-freqH);

	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				double iun=DIRECT_MULTIDIM_ELEM(iu,n);
//				double iun = *ptriun;
				double un=1.0/iun;
				if (freqH<=un && un<=freq)
				{
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
//					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= DIRECT_MULTIDIM_ELEM(conefilter, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= 0.5*(1+cos((un-freq)*ideltal));//H;
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = -J;
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(fftVRiesz, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= iun;
//					DIRECT_MULTIDIM_ELEM(coneVol, n) = DIRECT_MULTIDIM_ELEM(conefilter, n);
				} else if (un>freq)
				{
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
//					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= DIRECT_MULTIDIM_ELEM(conefilter, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = -J;
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(fftVRiesz, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= iun;
//					DIRECT_MULTIDIM_ELEM(coneVol, n) =  real(DIRECT_MULTIDIM_ELEM(myfftV, n)*conj(DIRECT_MULTIDIM_ELEM(myfftV, n)));
				}
				++n;
			}
		}
	}


//	#ifdef DEBUG_DIR
////	if ( (count == 0) )
////	{
//		Image<double> direction;
//		direction = coneVol;
//		direction.write(formatString("cone_%i_%i.vol", dir+1, count));
////	}
//	#endif

	transformer_inv.inverseFourierTransform(fftVRiesz, amplitude);

//	#ifdef DEBUG_DIR
//		Image<double> filteredvolume;
//		filteredvolume = VRiesz;
//		filteredvolume.write(formatString("Volumen_filtrado_%i_%i.vol", dir+1,count));
//	#endif


//	amplitude.resizeNoCopy(VRiesz);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) *= DIRECT_MULTIDIM_ELEM(amplitude,n);


	// Calculate first component of Riesz vector
	double ux;
	n=0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				ux = VEC_ELEM(freq_fourier,j);
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = ux*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
	}

	// Calculate second and third component of Riesz vector
	n=0;
	double uy, uz;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier,k);
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier,i);
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = uz*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = uy*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	transformer_inv.inverseFourierTransform(fftVRiesz_aux, VRiesz);


//	amplitude.setXmippOrigin();
	int z_size = ZSIZE(amplitude);
	int siz = z_size*0.5;

	double limit_radius = (siz-N_smoothing);
	n=0;
	for(int k=0; k<z_size; ++k)
	{
		uz = (k - siz);
		uz *= uz;
		for(int i=0; i<z_size; ++i)
		{
			uy = (i - siz);
			uy *= uy;
			for(int j=0; j<z_size; ++j)
			{
				ux = (j - siz);
				ux *= ux;
				DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
				DIRECT_MULTIDIM_ELEM(amplitude,n)=sqrt(DIRECT_MULTIDIM_ELEM(amplitude,n));
				double radius = sqrt(ux + uy + uz);
				if ((radius>=limit_radius) && (radius<=siz))
					DIRECT_MULTIDIM_ELEM(amplitude, n) *= 0.5*(1+cos(PI*(limit_radius-radius)/(N_smoothing)));
				else if (radius>siz)
					DIRECT_MULTIDIM_ELEM(amplitude, n) = 0;
				++n;
			}
		}
	}

	//TODO: change (k - z_size*0.5)

//		#ifdef MONO_AMPLITUDE
//		Image<double> saveImg2;
//		saveImg2 = amplitude;
//		if (fnDebug.c_str() != "")
//		{
//			FileName iternumber = formatString("smoothed_volume_%i_%i.vol", dir+1, count);
//			saveImg2.write(fnDebug+iternumber);
//		}
//		saveImg2.clear();
//		#endif


	transformer_inv.FourierTransform(amplitude, fftVRiesz, false);

	double raised_w = PI/(freqL-freq);

	n=0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftVRiesz)
	{
		double un=1.0/DIRECT_MULTIDIM_ELEM(iu,n);
		if ((freqL)>=un && un>=freq)
		{
			DIRECT_MULTIDIM_ELEM(fftVRiesz,n) *= 0.5*(1 + cos(raised_w*(un-freq)));
		}
		else
		{
			if (un>freqL)
			{
				DIRECT_MULTIDIM_ELEM(fftVRiesz,n) = 0;
			}
		}
	}

	transformer_inv.inverseFourierTransform();

//	#ifdef MONO_AMPLITUDE

//	if (fnDebug.c_str() != "")
//	{
//	Image<double> saveImg2;
//		saveImg2 = amplitude;
//		FileName iternumber = formatString("_Filtered_Amplitude_%i_%i.vol", dir+1, count);
//		saveImg2.write(fnDebug+iternumber);
//	}
//	#endif // DEBUG
}


void ProgResDir::defineCone(MultidimArray< std::complex<double> > &myfftV,
		MultidimArray< std::complex<double> > &conefilter, double rot, double tilt)
{
//	conefilter.initZeros(myfftV);
	conefilter = myfftV;
	// Filter the input volume and add it to amplitude

//	MultidimArray<double> conetest;
//	conetest.initZeros(myfftV);
//	#ifdef DEBUG_DIR
//	MultidimArray<double> coneVol;
//	coneVol.initZeros(iu);
//	#endif

	double x_dir, y_dir, z_dir;

	x_dir = sin(tilt*PI/180)*cos(rot*PI/180);
	y_dir = sin(tilt*PI/180)*sin(rot*PI/180);
	z_dir = cos(tilt*PI/180);

//	double ang_con = 10*PI/180;
	double ang_con = 15*PI/180;

	double uz, uy, ux;
	long n = 0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier,k);
		uz *= z_dir;
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier,i);
			uy *= y_dir;
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				double iun=DIRECT_MULTIDIM_ELEM(iu,n);
				ux = VEC_ELEM(freq_fourier,j);
				ux *= x_dir;

				//BE CAREFULL with the order
				//double dotproduct = (uy*x_dir + ux*y_dir + uz*z_dir)*iun;
				iun *= (ux + uy + uz);
				double acosine = acos(fabs(iun));
//				DIRECT_MULTIDIM_ELEM(conetest, n) = log(real(conj(DIRECT_MULTIDIM_ELEM(myfftV, n))*DIRECT_MULTIDIM_ELEM(myfftV, n)));
				if (acosine>ang_con)
				{
					DIRECT_MULTIDIM_ELEM(conefilter, n) = 0;
//					DIRECT_MULTIDIM_ELEM(conetest, n) = 0;
				}
/*
				//4822.53 mean a smoothed cone angle of 20 degrees
				double arg_exp = acosine*acosine*acosine*acosine*4822.53;
				DIRECT_MULTIDIM_ELEM(conefilter, n) *= exp(-arg_exp);
//				DIRECT_MULTIDIM_ELEM(conetest, n) = exp(-arg_exp);
 */
				++n;
			}
		}
	}
//
//	Image<double> saveImg2;
//	saveImg2 = conetest;
//	saveImg2.write("cono.vol");

}

void ProgResDir::diagSymMatrix3x3(Matrix2D<double> A,
					Matrix1D<double> &eigenvalues, Matrix2D<double> &eigenvectors)
{
	Matrix2D<double> B;
	B.initZeros(3,3);

	MAT_ELEM(B,0,0) = 1;
	MAT_ELEM(B,1,1) = 1;
	MAT_ELEM(B,2,2) = 1;

	generalizedEigs(A, B, eigenvalues, eigenvectors);
}

void ProgResDir::resolution2eval_(int &fourier_idx, double min_step,
								double &resolution, double &last_resolution,
								int &last_fourier_idx,
								double &freq, double &freqL, double &freqH,
								bool &continueIter, bool &breakIter, bool &doNextIteration)
{
	int volsize = ZSIZE(VRiesz);

	FFT_IDX2DIGFREQ(fourier_idx, volsize, freq);

	resolution = sampling/freq;
//	std::cout << "res = " << resolution << std::endl;
//	std::cout << "min_step = " << min_step << std::endl;
	if (resolution>8)
		min_step =1;



	if ( fabs(resolution - last_resolution)<min_step )
	{
		freq = sampling/(last_resolution-min_step);
		DIGFREQ2FFT_IDX(freq, volsize, fourier_idx);
		FFT_IDX2DIGFREQ(fourier_idx, volsize, freq);

		if (fourier_idx == last_fourier_idx)
		{
			continueIter = true;
			++fourier_idx;
			return;
		}
	}

	resolution = sampling/freq;
	last_resolution = resolution;

	double step = 0.05*resolution;

	double resolution_L, resolution_H;

	if ( step < min_step)
	{
		resolution_L = resolution - min_step;
		resolution_H = resolution + min_step;
	}
	else
	{
		resolution_L = 0.95*resolution;
		resolution_H = 1.05*resolution;
	}

	freqH = sampling/(resolution_H);
	freqL = sampling/(resolution_L);

	if (freqH>0.5 || freqH<0)
		freqH = 0.5;

	if (freqL>0.5 || freqL<0)
		freqL = 0.5;
	int fourier_idx_H, fourier_idx_L;

	DIGFREQ2FFT_IDX(freqH, volsize, fourier_idx_H);
	DIGFREQ2FFT_IDX(freqL, volsize, fourier_idx_L);

	if (fourier_idx_H == fourier_idx)
		fourier_idx_H = fourier_idx - 1;

	if (fourier_idx_L == fourier_idx)
		fourier_idx_L = fourier_idx + 1;

	FFT_IDX2DIGFREQ(fourier_idx_H, volsize, freqH);
	FFT_IDX2DIGFREQ(fourier_idx_L, volsize, freqL);

//	std::cout << "freq_H = " << freqH << std::endl;
//	std::cout << "freq_L = " << freqL << std::endl;

	if (freq>0.49 || freq<0)
	{
		std::cout << "Nyquist limit reached" << std::endl;
		breakIter = true;
		doNextIteration = false;
		return;
	}
	else
	{
		breakIter = false;
		doNextIteration = true;
	}
//	std::cout << "resolution = " << resolution << "  resolutionL = " <<
//				sampling/(freqL) << "  resolutionH = " << sampling/freqH
//				<< "  las_res = " << last_resolution << std::endl;
	last_fourier_idx = fourier_idx;
	++fourier_idx;
}


void ProgResDir::removeOutliers(Matrix2D<double> &anglesMat,
		Matrix2D<double> &resolutionMat)
{
	std::cout << "Removing outliers..." << std::endl;

	double x1, y1, z1, x2, y2, z2, distance, resolution, sigma,
				rot, tilt, threshold, sigma2, lastMinDistance;
	double meandistance = 0, distance_2 = 0;
	int numberdirections = angles.mdimx, N=0, count = 0;

	double criticalZ = icdf_gauss(significance);
	double threshold_gauss;
	double counter = 0;

	double ang = 20.0;

	Matrix2D<double> neigbour_dir;

	for (int k = 0; k<NVoxelsOriginalMask; ++k)
	{
		meandistance = 0;
		sigma = 0;
		distance_2 = 0;
		counter = 0;

//		std::vector<double> neighbours;
		neigbour_dir.initZeros(numberdirections, 2);
		//Computing closest neighbours and its mean distance
		for (int i = 0; i<numberdirections; ++i)
		{
//			if ((k == 201311) || (k == 201312) || (k == 283336) || (k == 324353) || (k == 324362) || (k == 324512))
//			{
//				std::cout << k << " " << MAT_ELEM(resolutionMat, i, k) << " " << MAT_ELEM(trigProducts, 0, i) << "  " <<
//						MAT_ELEM(trigProducts, 1, i) << " " << MAT_ELEM(trigProducts, 2, i) << ";" << std::endl;
//			}

			double resi = MAT_ELEM(resolutionMat, i, k);
			if (resi>0)
			{
				x1 = MAT_ELEM(trigProducts, 0, i);
				y1 = MAT_ELEM(trigProducts, 1, i);
				z1 = MAT_ELEM(trigProducts, 2, i);

				for (int j = 0; j<numberdirections; ++j)
				{
					if (i != j)
					{
						x2 = MAT_ELEM(trigProducts, 0, j);
						y2 = MAT_ELEM(trigProducts, 1, j);
						z2 = MAT_ELEM(trigProducts, 2, j);
						double resj = MAT_ELEM(resolutionMat, j, k);
						if (resj>0)
						{
							distance = (180/PI)*acos(x1*x2 + y1*y2 + z1*z2);

							if (distance < ang)
							{
								x2 *= resj;
								y2 *= resj;
								z2 *= resj;
								distance = sqrt( (resi*x1-x2)*(resi*x1-x2) + (resi*y1-y2)*(resi*y1-y2) + (resi*z1-z2)*(resi*z1-z2) );
								MAT_ELEM(neigbour_dir, i, 0) += distance;
								MAT_ELEM(neigbour_dir, i, 1) += 1;
//								neighbours.push_back(distance);
								meandistance += distance;
								++counter;
								sigma += distance*distance;
							}
						}
					}
				}
			}
		}

		double thresholdDirection;

		meandistance= meandistance/counter;
		sigma = sigma/counter - meandistance*meandistance;

//		std::sort(neighbours.begin(), neighbours.end());
//		thresholdDirection = neighbours[size_t(neighbours.size()*significance)];
		threshold_gauss = meandistance + criticalZ*sqrt(sigma);

//		neighbours.clear();

		//A direction is an outlier if is significative higher than overal distibution

		for (int i = 0; i<numberdirections; ++i)
		{
			double meandistance;
			if (MAT_ELEM(neigbour_dir, i, 0)>0)
			{
				meandistance = MAT_ELEM(neigbour_dir, i, 0)/MAT_ELEM(neigbour_dir, i, 1);

				if ((meandistance>threshold_gauss) || MAT_ELEM(neigbour_dir, i, 0) <= 1)
				{
					MAT_ELEM(resolutionMat, i, k)=-1;
				}
			}

		}

//		if ((k == 201311) || (k == 201312) || (k == 283336) || (k == 324353) || (k == 324362) || (k == 324512))
//			std::cout << "threshold_gauss--------------=" << threshold_gauss << std::endl;

	}
}



void ProgResDir::ellipsoidFitting(Matrix2D<double> &anglesMat,
									Matrix2D<double> &resolutionMat,
									Matrix2D<double> &axis)
{

	std::cout << "FITTIG" << std::endl;
	double x, y, z, a, b, c, resolution, rot, tilt;
	int numberdirections = angles.mdimx;
	std::vector<double> list_distances;

	//std::cout << "xrows = " << xrows << std::endl;

	//MAT_ELEM(resolutionMat, direccion, resolucion)

	Matrix2D<double> ellipMat;
	int dimMatrix = 0;
	size_t mycounter;
	Matrix2D<double> pseudoinv, quadricMatrix;
	Matrix1D<double> onesVector, leastSquares, residuals, residualssorted;
	Matrix2D<double> eigenvectors;
	Matrix1D<double> eigenvalues;
	//rows 1 2 3 (length axis a b c- where a is the smallest one)
	//rows 4 5 6 x y z coordinates of the first eigenvector
	//rows 7 8 9 x y z coordinates of the second eigenvector
	//rows 10 11 12 x y z coordinates of the third eigenvector
	axis.initZeros(12, NVoxelsOriginalMask);

	quadricMatrix.initZeros(3,3);
	for (int k = 0; k<NVoxelsOriginalMask; ++k)
	{
		dimMatrix = 0;
		for (int i = 0; i<numberdirections; ++i)
		{
			if (MAT_ELEM(resolutionMat, i, k) > 0)
				++dimMatrix;
		}

		ellipMat.initZeros(dimMatrix, 6);
		mycounter = 0; //It is required to store the matrix ellipMat
		for (int i = 0; i<numberdirections; ++i)
		{
			resolution = MAT_ELEM(resolutionMat, i, k);

			if (resolution>0)
			{
				x = resolution*MAT_ELEM(trigProducts, 0, i);
				y = resolution*MAT_ELEM(trigProducts, 1, i);
				z = resolution*MAT_ELEM(trigProducts, 2, i);

				MAT_ELEM(ellipMat, mycounter, 0) = x*x;
				MAT_ELEM(ellipMat, mycounter, 1) = y*y;
				MAT_ELEM(ellipMat, mycounter, 2) = z*z;
				MAT_ELEM(ellipMat, mycounter, 3) = 2*x*y;
				MAT_ELEM(ellipMat, mycounter, 4) = 2*x*z;
				MAT_ELEM(ellipMat, mycounter, 5) = 2*y*z;
				++mycounter;
			}
		}


		ellipMat.inv(pseudoinv);

		onesVector.initConstant(mycounter, 1.0);
		leastSquares = pseudoinv*onesVector;

		//Removing outliers
		residuals = ellipMat*leastSquares - onesVector;
		residualssorted = residuals.sort();

		double threshold_plus = VEC_ELEM(residualssorted, size_t(residualssorted.size()*significance));
		double threshold_minus = VEC_ELEM(residualssorted, size_t(residualssorted.size()*(1.0-significance)));

		mycounter = 0;
		size_t ellipsoidcounter = 0;
		for (int i = 0; i<numberdirections; ++i)
		{
			resolution = MAT_ELEM(resolutionMat, i, k);

			if (resolution>0)
			{
				if ( (VEC_ELEM(residuals, mycounter) > threshold_plus) ||
						(VEC_ELEM(residuals, mycounter) < threshold_minus) )
				{
					MAT_ELEM(resolutionMat, i, k) = -1;
				}
				else
				{
					++ellipsoidcounter;
				}
				++mycounter;
			}
		}

		ellipMat.initZeros(ellipsoidcounter, 6);
		mycounter = 0; //It is required to store the matrix ellipMat
		for (int i = 0; i<numberdirections; ++i)
		{
			resolution = MAT_ELEM(resolutionMat, i, k);
//			if ((k == 201311) || (k == 201312) || (k == 283336) || (k == 324353) || (k == 324362) || (k == 324512))
//			{
//				std::cout << k << " " << resolution << " " << MAT_ELEM(trigProducts, 0, i)
//						<< " " << MAT_ELEM(trigProducts, 1, i)
//						<< " " << MAT_ELEM(trigProducts, 2, i) << ";" << std::endl;
//			}

			if (resolution>0)
			{
				x = resolution*MAT_ELEM(trigProducts, 0, i);
				y = resolution*MAT_ELEM(trigProducts, 1, i);
				z = resolution*MAT_ELEM(trigProducts, 2, i);

				MAT_ELEM(ellipMat, mycounter, 0) = x*x;
				MAT_ELEM(ellipMat, mycounter, 1) = y*y;
				MAT_ELEM(ellipMat, mycounter, 2) = z*z;
				MAT_ELEM(ellipMat, mycounter, 3) = 2*x*y;
				MAT_ELEM(ellipMat, mycounter, 4) = 2*x*z;
				MAT_ELEM(ellipMat, mycounter, 5) = 2*y*z;
				++mycounter;
			}
		}

		// defining ellipsoid

		ellipMat.inv(pseudoinv);

		onesVector.initConstant(mycounter, 1.0);
		leastSquares = pseudoinv*onesVector;


		MAT_ELEM(quadricMatrix, 0, 0) = VEC_ELEM(leastSquares, 0);
		MAT_ELEM(quadricMatrix, 0, 1) = VEC_ELEM(leastSquares, 3);
		MAT_ELEM(quadricMatrix, 0, 2) = VEC_ELEM(leastSquares, 4);
		MAT_ELEM(quadricMatrix, 1, 0) = VEC_ELEM(leastSquares, 3);
		MAT_ELEM(quadricMatrix, 1, 1) = VEC_ELEM(leastSquares, 1);
		MAT_ELEM(quadricMatrix, 1, 2) = VEC_ELEM(leastSquares, 5);
		MAT_ELEM(quadricMatrix, 2, 0) = VEC_ELEM(leastSquares, 4);
		MAT_ELEM(quadricMatrix, 2, 1) = VEC_ELEM(leastSquares, 5);
		MAT_ELEM(quadricMatrix, 2, 2) = VEC_ELEM(leastSquares, 2);

		diagSymMatrix3x3(quadricMatrix, eigenvalues, eigenvectors);

		if (VEC_ELEM(eigenvalues, 0)<0)
		{//This is de the vectorial product
			VEC_ELEM(eigenvalues, 0) = VEC_ELEM(eigenvalues, 1);
			MAT_ELEM(axis,3, k) = MAT_ELEM(eigenvectors,0,1)*MAT_ELEM(eigenvectors,2,2)-
									MAT_ELEM(eigenvectors,2,1)*MAT_ELEM(eigenvectors,1,2);
			MAT_ELEM(axis,4, k) = MAT_ELEM(eigenvectors,2,1)*MAT_ELEM(eigenvectors,0,2)-
									MAT_ELEM(eigenvectors,0,1)*MAT_ELEM(eigenvectors,2,2);
			MAT_ELEM(axis,5, k) = MAT_ELEM(eigenvectors,0,1)*MAT_ELEM(eigenvectors,1,2)-
									MAT_ELEM(eigenvectors,1,1)*MAT_ELEM(eigenvectors,0,2);
		}

		a = 1/sqrt(VEC_ELEM(eigenvalues, 0));
		b = 1/sqrt(VEC_ELEM(eigenvalues, 1));
		c = 1/sqrt(VEC_ELEM(eigenvalues, 2));

//		if ((k == 201311) || (k == 201312) || (k == 283336) || (k == 324353) || (k == 324362) || (k == 324512))
//		{
//			std::cout << "a = " << a << std::endl;
//			std::cout << "b = " << b << std::endl;
//			std::cout << "c = " << c << std::endl;
//			std::cout << "=--------------=" << std::endl;
//		}

		MAT_ELEM(axis,0, k) = a;
		MAT_ELEM(axis,1, k) = b;
		MAT_ELEM(axis,2, k) = c;

		MAT_ELEM(axis,3, k) = MAT_ELEM(eigenvectors,0,0);
		MAT_ELEM(axis,4, k) = MAT_ELEM(eigenvectors,1,0);
		MAT_ELEM(axis,5, k) = MAT_ELEM(eigenvectors,2,0);

		MAT_ELEM(axis,6, k) = MAT_ELEM(eigenvectors,0,1);
		MAT_ELEM(axis,7, k) = MAT_ELEM(eigenvectors,1,1);
		MAT_ELEM(axis,8, k) = MAT_ELEM(eigenvectors,2,1);

		MAT_ELEM(axis,9, k) = MAT_ELEM(eigenvectors,0,2);
		MAT_ELEM(axis,10, k) = MAT_ELEM(eigenvectors,1,2);
		MAT_ELEM(axis,11, k) = MAT_ELEM(eigenvectors,2,2);
	}
}


void ProgResDir::radialAverageInMask(MultidimArray<int> &mask,
		MultidimArray<double> &inputVol_1, MultidimArray<double> &inputVol_2,
		MultidimArray<double> &inputVol_3, MultidimArray<double> &inputVol_4,
		MultidimArray<double> &inputVol_5, MetaData &md)
{
		double u_inf, u_sup, u;

		int step = 1;

		double N, NN;
		MultidimArray<double> radialAvg(XSIZE(inputVol_1)*0.5);
		int uk, uj, ui;

		int siz = XSIZE(inputVol_1);
		size_t objId;

		inputVol_1.setXmippOrigin();
		inputVol_2.setXmippOrigin();
		inputVol_3.setXmippOrigin();
		inputVol_4.setXmippOrigin();
		inputVol_5.setXmippOrigin();
		MultidimArray<double> zVolume;
		zVolume.initZeros(inputVol_1);
		zVolume.setXmippOrigin();

		MultidimArray<int> &pMask = mask;
		pMask.setXmippOrigin();

		Matrix2D<double> std_mean_Radial_1, std_mean_Radial_2, std_mean_Radial_3,
						std_mean_Radial_4, std_mean_Radial_5;

		std_mean_Radial_1.initZeros(2, (size_t) siz*0.5 + 1);
		std_mean_Radial_2.initZeros(2, (size_t) siz*0.5 + 1);
		std_mean_Radial_3.initZeros(2, (size_t) siz*0.5 + 1);
		std_mean_Radial_4.initZeros(2, (size_t) siz*0.5 + 1);
		std_mean_Radial_5.initZeros(2, (size_t) siz*0.5 + 1);

		for(size_t kk=1; kk<siz*0.5; ++kk)
		{
			double cum_mean_1 = 0;
			double cum_mean_2 = 0;
			double cum_mean_3 = 0;
			double cum_mean_4 = 0;
			double cum_mean_5 = 0;

			double cum2_mean_1 = 0;
			double cum2_mean_2 = 0;
			double cum2_mean_3 = 0;
			double cum2_mean_4 = 0;
			double cum2_mean_5 = 0;

			N = 0;
			u_sup = kk + step;
			u_inf = kk - step;

			FOR_ALL_ELEMENTS_IN_ARRAY3D(pMask)
			{
				 if (A3D_ELEM(pMask, k, i, j)>0)
				{
					u = sqrt(k*k + i*i + j*j);
					if ((u<u_sup) && (u>=u_inf))
					{
						double aux;
						aux = A3D_ELEM(inputVol_1, k, i, j);
						cum_mean_1 += aux;
						cum2_mean_1 += aux*aux;
						aux = A3D_ELEM(inputVol_2, k, i, j);
						cum_mean_2 += aux;
						cum2_mean_2 += aux*aux;
						aux = A3D_ELEM(inputVol_3, k, i, j);
						cum_mean_3 += aux;
						cum2_mean_3 += aux*aux;
						aux = A3D_ELEM(inputVol_4, k, i, j);
						cum_mean_4 += aux;
						cum2_mean_4 += aux*aux;
						aux = A3D_ELEM(inputVol_5, k, i, j);
						cum_mean_5 += aux;
						cum2_mean_5 += aux*aux;

						N = N + 1;
					}
				 }
			}

			double sigma1, sigma2, sigma3, sigma4, sigma5;


			objId = md.addObject();
//			if ((cum_mean_1==0) || (cum_mean_2==0) || (cum_mean_3==0) || (cum_mean_4==0) || (cum_mean_5==0))
//			{
//				md.setValue(MDL_IDX, kk, objId);
//				md.setValue(MDL_VOLUME_SCORE1, cum_mean_1, objId);
//				md.setValue(MDL_VOLUME_SCORE2, cum_mean_2, objId);
//				md.setValue(MDL_VOLUME_SCORE3, cum_mean_3, objId);
//				md.setValue(MDL_VOLUME_SCORE4, cum_mean_4, objId);
//				md.setValue(MDL_AVG, cum_mean_5, objId);
//			}
//			else
//			{
			if ((cum_mean_1>0) || (cum_mean_2>0) || (cum_mean_3>0) || (cum_mean_4>0) || (cum_mean_5>0))
			{
				cum_mean_1 = (cum_mean_1/N);
				cum_mean_2 = (cum_mean_2/N);
				cum_mean_3 = (cum_mean_3/N);
				cum_mean_4 = (cum_mean_4/N);
				cum_mean_5 = (cum_mean_5/N);

				MAT_ELEM(std_mean_Radial_1,1, kk) = cum_mean_1;
				MAT_ELEM(std_mean_Radial_2,1, kk) = cum_mean_2;
				MAT_ELEM(std_mean_Radial_3,1, kk) = cum_mean_3;
				MAT_ELEM(std_mean_Radial_4,1, kk) = cum_mean_4;
				MAT_ELEM(std_mean_Radial_5,1, kk) = cum_mean_5;


				md.setValue(MDL_IDX, kk, objId);
				md.setValue(MDL_VOLUME_SCORE1, cum_mean_1, objId);
				md.setValue(MDL_VOLUME_SCORE2, cum_mean_2, objId);
				md.setValue(MDL_VOLUME_SCORE3, cum_mean_3, objId);
				md.setValue(MDL_VOLUME_SCORE4, cum_mean_4, objId);
				md.setValue(MDL_AVG, cum_mean_5, objId);

				MAT_ELEM(std_mean_Radial_1,0, kk) = sqrt(cum2_mean_1/(N) - cum_mean_1*cum_mean_1);
				MAT_ELEM(std_mean_Radial_2,0, kk) = sqrt(cum2_mean_2/(N) - cum_mean_2*cum_mean_2);
				MAT_ELEM(std_mean_Radial_3,0, kk) = sqrt(cum2_mean_3/(N) - cum_mean_3*cum_mean_3);
				MAT_ELEM(std_mean_Radial_4,0, kk) = sqrt(cum2_mean_4/(N) - cum_mean_4*cum_mean_4+0.001);
				MAT_ELEM(std_mean_Radial_5,0, kk) = sqrt(cum2_mean_5/(N) - cum_mean_5*cum_mean_5);

//				std::cout << "cum2_mean_4/(N) = " << cum2_mean_4/(N)  << " " << "cum_mean_4*cum_mean_4) = " << cum_mean_4*cum_mean_4 << std::endl;
// 				std::cout << "MAT_ELEM(std_mean_Radial_1,0, kk) = " << MAT_ELEM(std_mean_Radial_1,0, kk) << " " << MAT_ELEM(std_mean_Radial_2,0, kk) << " " << MAT_ELEM(std_mean_Radial_3,0, kk) << " " << MAT_ELEM(std_mean_Radial_4,0, kk) << " " << MAT_ELEM(std_mean_Radial_5,0, kk)<< std::endl;
//				std::cout << "MAT_ELEM(std_mean_Radial_1,1, kk) = " << MAT_ELEM(std_mean_Radial_1,1, kk) << " "  << MAT_ELEM(std_mean_Radial_2,1, kk) << " " << MAT_ELEM(std_mean_Radial_3,1, kk) << " " << MAT_ELEM(std_mean_Radial_4,1, kk) << " " << MAT_ELEM(std_mean_Radial_5,1, kk)<< std::endl;
			}


			double lastz;
			FOR_ALL_ELEMENTS_IN_ARRAY3D(pMask)
			{
				 if (A3D_ELEM(pMask, k, i, j)>0)
				{
					lastz = -1;
					u = sqrt(k*k + i*i + j*j);
					if ((u<u_sup) && (u>=u_inf) && (MAT_ELEM(std_mean_Radial_1,1, kk)>0))
					{
						double z, aux;
						aux = abs((A3D_ELEM(inputVol_1, k, i, j) - MAT_ELEM(std_mean_Radial_1,1, kk))/MAT_ELEM(std_mean_Radial_1,0, kk)) +  + 0.002;
						if (aux > lastz)
							lastz = aux;
						aux = abs((A3D_ELEM(inputVol_2, k, i, j) - MAT_ELEM(std_mean_Radial_2,1, kk))/MAT_ELEM(std_mean_Radial_2,0, kk))  + 0.002;
						if (aux > lastz)
							lastz = aux;
						aux = abs((A3D_ELEM(inputVol_3, k, i, j) - MAT_ELEM(std_mean_Radial_3,1, kk))/MAT_ELEM(std_mean_Radial_3,0, kk))  + 0.002;
						if (aux > lastz)
							lastz = aux;
//						aux = abs((A3D_ELEM(inputVol_4, k, i, j) - MAT_ELEM(std_mean_Radial_4,1, kk))/MAT_ELEM(std_mean_Radial_4,0, kk));
//						if (aux > lastz)
//							lastz = aux;
						aux = abs((A3D_ELEM(inputVol_5, k, i, j) - MAT_ELEM(std_mean_Radial_5,1, kk))/MAT_ELEM(std_mean_Radial_5,0, kk))  + 0.002;
						if (aux > lastz)
							lastz = aux;

						if (lastz>5.0) //This line considers zscores higher than 5 as 5(saturated at 5sigma)
							lastz = 5;

						A3D_ELEM(zVolume, k, i, j) = lastz;
					}
				 }
			}
		}

		Image<double> zVolumesave;
		zVolumesave = zVolume;
		zVolumesave.write(fnZscore);
}

void ProgResDir::radialAzimuthalResolution(Matrix2D<double> &resolutionMat,
		MultidimArray<int> &pmask,
		MultidimArray<double> &radial,
		MultidimArray<double> &azimuthal,
//		MultidimArray<double> &meanResolution,
		MultidimArray<double> &lowestResolution,
		MultidimArray<double> &highestResolution,
		MultidimArray<double> &doaResolution_1,
		MultidimArray<double> &doaResolution_2,
		double &radial_Thr, double &azimuthal_Thr,
		MetaData &mdprefDirs)
{

	radial.initZeros(pmask);
	azimuthal.initZeros(pmask);
	lowestResolution.initZeros(pmask);
	highestResolution.initZeros(pmask);
	doaResolution_1.initZeros(pmask);
	doaResolution_2.initZeros(pmask);

	double radial_angle = 45*PI/180;
	double azimuthal_resolution = 0;
	double radial_resolution = 0;
	double azimuthal_angle = 70*PI/180;
	double resolution, dotproduct, x, y, z, iu, arcos;
	int xrows = angles.mdimx;
	int idx = 0;

	double count_radial, count_azimuthal;

	Matrix1D<int> PrefferredDirHist, resolutionMeanVector;
	PrefferredDirHist.initZeros(xrows);
	resolutionMeanVector.initZeros(xrows);

//	meanResolution.initZeros(pmask);
	size_t objId;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(pmask)
	{
		//i defines the direction and k the voxel
		if (A3D_ELEM(pmask,k,i,j) > 0 )
		{
			iu = 1/sqrt(i*i + j*j + k*k);
			count_radial = 0;
			count_azimuthal = 0;
			std::vector<double> ResList;

			double lastRes = 100; //A non-sense value

			for (int ii = 0; ii<xrows; ++ii)
			{
				resolution = MAT_ELEM(resolutionMat, ii, idx);

				if (resolution>0)
				{
					ResList.push_back(resolution);
					x = MAT_ELEM(trigProducts, 0, ii);
					y = MAT_ELEM(trigProducts, 1, ii);
					z = MAT_ELEM(trigProducts, 2, ii);

					dotproduct = (x*i + y*j + z*k)*iu;
					arcos = acos(fabs(dotproduct));
					if (arcos>=azimuthal_angle)
					{
						count_azimuthal = count_azimuthal + 1;
						azimuthal_resolution += resolution;
					}
					if (arcos<=radial_angle)
					{
						count_radial = count_radial + 1;
						radial_resolution += resolution;
					}

				}

			}

//			std::cout << "count_radial = " << count_radial << std::endl;
//			std::cout << "count_azimuthal = " << count_azimuthal << std::endl;
//			std::cout << "  " << std::endl;

//			A3D_ELEM(meanResolution,k,i,j) = meanRes[(size_t) floor(0.5*meanRes.size())];
			std::sort(ResList.begin(),ResList.end());

			double Mres, mres, medianResol;

			Mres = ResList[ (size_t) floor(0.95*ResList.size()) ];
			A3D_ELEM(lowestResolution,k,i,j) = Mres;

			mres = ResList[ (size_t) floor(0.03*ResList.size()) ];

			medianResol = ResList[ (size_t) floor(0.5*ResList.size()) ];

			A3D_ELEM(highestResolution,k,i,j) = mres;

			A3D_ELEM(doaResolution_1,k,i,j) = ( (Mres - mres)/(Mres + mres) );

			A3D_ELEM(doaResolution_2,k,i,j) = 0.5*( (Mres + mres) );

			ResList.clear();

			//Prefferred directions

			for (int ii = 0; ii<xrows; ++ii)
			{
				resolution = MAT_ELEM(resolutionMat, ii, idx);

				if (resolution>0)
				{
					if ((mres>(resolution-0.1)) && (mres<(resolution+0.1)))
					{
						VEC_ELEM(PrefferredDirHist,ii) += 1;
						VEC_ELEM(resolutionMeanVector,ii) += resolution;
					}
				}

			}
			++idx;
		}

		if (count_radial<1)
			A3D_ELEM(radial,k,i,j) = A3D_ELEM(doaResolution_2,k,i,j);
		else
			A3D_ELEM(radial,k,i,j) = radial_resolution/count_radial;

		if (count_azimuthal<1)
			A3D_ELEM(azimuthal,k,i,j) = A3D_ELEM(doaResolution_2,k,i,j);
		else
			A3D_ELEM(azimuthal,k,i,j) = azimuthal_resolution/count_azimuthal;


		azimuthal_resolution = 0;
		radial_resolution = 0;
	}


	for (size_t ii = 0; ii<xrows; ++ii)
	{
		objId = mdprefDirs.addObject();
		size_t con;
		con = (size_t) VEC_ELEM(PrefferredDirHist,ii);
//		std::cout << ii << " " << con << std::endl;
		double rot = MAT_ELEM(angles, 0, ii);
		double tilt = MAT_ELEM(angles, 1, ii);

		if (tilt<0)
		{
			tilt = abs(tilt);
			rot = rot + 180;
		}
		double meanRes;

		meanRes = VEC_ELEM(resolutionMeanVector,ii)/((double) VEC_ELEM(PrefferredDirHist,ii));

		mdprefDirs.setValue(MDL_ANGLE_ROT, rot, objId);
		mdprefDirs.setValue(MDL_ANGLE_TILT, tilt, objId);
		mdprefDirs.setValue(MDL_WEIGHT, (double) con, objId);
		mdprefDirs.setValue(MDL_RESOLUTION_FREQ, meanRes, objId);
		mdprefDirs.setValue(MDL_X, (double) ii, objId);
		mdprefDirs.setValue(MDL_COUNT, con, objId);
	}
	mdprefDirs.write(fnprefMin);

	std::vector<double> radialList, azimuthalList;

	FOR_ALL_ELEMENTS_IN_ARRAY3D(radial)
	{
		//i defines the direction and k the voxel
		if (A3D_ELEM(pmask,k,i,j) > 0 )
		{
			radialList.push_back(A3D_ELEM(radial,k,i,j));
			azimuthalList.push_back(A3D_ELEM(azimuthal,k,i,j));
		}
	}

	std::sort(radialList.begin(),radialList.end());
	std::sort(azimuthalList.begin(),azimuthalList.end());

	radial_Thr = radialList[(size_t) floor(radialList.size()*0.9)];
	azimuthal_Thr = azimuthalList[(size_t) floor(azimuthalList.size()*0.9)];
}

//TODO: change this function to be more efficient
double ProgResDir::firstMonoResEstimation(MultidimArray< std::complex<double> > &myfftV,
		double freq, double freqH, MultidimArray<double> &amplitude)
{
	fftVRiesz.initZeros(myfftV);
	amplitude.resizeNoCopy(VRiesz);
	std::complex<double> J(0,1);

	// Filter the input volume and add it to amplitude
	long n=0;
	double iw=1.0/freq;
	double iwl=1.0/freqH;
	double ideltal=PI/(freq-freqH);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(myfftV)
	{
		double iun=DIRECT_MULTIDIM_ELEM(iu,n);
		double un=1.0/iun;
		if (freqH<=un && un<=freq)
		{
			//double H=0.5*(1+cos((un-w1)*ideltal));
			DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
			DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= 0.5*(1+cos((un-freq)*ideltal));//H;
		} else if (un>freq)
			DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n)=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	// Calculate first component of Riesz vector
	fftVRiesz.initZeros(myfftV);
	double uz, uy, ux;
	n=0;

	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				ux = VEC_ELEM(freq_fourier,j);
				double iun=DIRECT_MULTIDIM_ELEM(iu,n);
				double un=1.0/iun;
				if (freqH<=un && un<=freq)
				{
					//double H=0.5*(1+cos((un-w1)*ideltal));
					//DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = (-J*ux*iun)*DIRECT_MULTIDIM_ELEM(myfftV, n);
					//DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *=0.5*(1+cos((un-w1)*ideltal));//H;
					//Next lines are an optimization of the commented ones
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= J;
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= -ux*iun*0.5*(1+cos((un-freq)*ideltal));//H;
				} else if (un>freq)
				{
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = (-ux*iun)*DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= J;
				}
				++n;
			}
		}
	}
	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	// Calculate second component of Riesz vector
	fftVRiesz.initZeros(myfftV);
	n=0;

	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier,i);
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				double iun=DIRECT_MULTIDIM_ELEM(iu,n);
				double un=1.0/iun;
				if (freqH<=un && un<=freq)
				{
					//double H=0.5*(1+cos((un-w1)*ideltal));
					//DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = (-J*uy*iun)*DIRECT_MULTIDIM_ELEM(myfftV, n);
					//DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *=0.5*(1+cos((un-w1)*ideltal));//H;
					//Next lines are an optimization of the commented ones
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= J;
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= -uy*iun*0.5*(1+cos((un-freq)*ideltal));//H;
				} else if (un>freq)
				{
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = (-uy*iun)*DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= J;
				}
				++n;
			}
		}
	}
	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	// Calculate third component of Riesz vector
	fftVRiesz.initZeros(myfftV);
	n=0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier,k);
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				double iun=DIRECT_MULTIDIM_ELEM(iu,n);
				double un=1.0/iun;
				if (freqH<=un && un<=freq)
				{
					//double H=0.5*(1+cos((un-w1)*ideltal));
					//DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = (-J*uz*iun)*DIRECT_MULTIDIM_ELEM(myfftV, n);
					//DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= 0.5*(1+cos((un-w1)*ideltal));//H;
					//Next lines are an optimization of the commented ones
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= J;
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= -uz*iun*0.5*(1+cos((un-freq)*ideltal));//H;
				} else if (un>freq)
				{
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = (-uz*iun)*DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= J;
				}
				++n;
			}
		}
	}
	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
		DIRECT_MULTIDIM_ELEM(amplitude,n)=sqrt(DIRECT_MULTIDIM_ELEM(amplitude,n));
	}

	// Low pass filter the monogenic amplitude
	// Prepare low pass filter
	FourierFilter lowPassFilter, FilterBand;
	lowPassFilter.FilterShape = RAISED_COSINE;
	lowPassFilter.raised_w = 0.01;
	lowPassFilter.do_generate_3dmask = false;
	lowPassFilter.FilterBand = LOWPASS;
	lowPassFilter.w1 = freq;
	amplitude.setXmippOrigin();
	lowPassFilter.applyMaskSpace(amplitude);

	double sumS=0, sumS2=0, sumN=0, sumN2=0, NN = 0, NS = 0;
	MultidimArray<int> &pMask = mask();
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		double amplitudeValue=DIRECT_MULTIDIM_ELEM(amplitude, n);
		if (DIRECT_MULTIDIM_ELEM(pMask, n)==0)
		{
			sumN  += amplitudeValue;
			sumN2 += amplitudeValue*amplitudeValue;
			++NN;
		}
	}

	double mean_noise = sumN/NN;
	return mean_noise;

}

void ProgResDir::run()
{
	produceSideInfo();
	bool continueIter = false, breakIter = false;
	double criticalZ=icdf_gauss(significance);

	double step;
	step = res_step;

	std::cout << "Analyzing directions " << std::endl;
	std::cout << "maxRes = " << maxRes << std::endl;
	std::cout << "minRes = " << minRes << std::endl;
	std::cout << "N_freq = " << N_freq << std::endl;
	std::cout << "step = " << step << std::endl;
	std::cout << "criticalZ = " << criticalZ << std::endl;


	double w, wH;
	int volsize = ZSIZE(VRiesz);

	//Checking with MonoRes at 50A;
	int aux_idx;

	if (maxRes>18)
	{
		DIGFREQ2FFT_IDX(sampling/18, volsize, aux_idx);

		FFT_IDX2DIGFREQ(aux_idx, volsize, w);
		FFT_IDX2DIGFREQ(aux_idx+1, volsize, wH); //Frequency chosen for a first estimation
	}
	else
	{
		FFT_IDX2DIGFREQ(3, volsize, w);
		FFT_IDX2DIGFREQ(4, volsize, w);
		aux_idx = 3;
	}
	std::cout << "fourier idx = " << aux_idx << std::endl;
	std::cout << "Calling MonoRes core as a first estimation at " << sampling/w << "A." << std::endl;

	MultidimArray<double> amplitudeMS;
	double AvgNoise;
	AvgNoise = firstMonoResEstimation(fftV, w, wH, amplitudeMS)/9.0;

	N_directions=angles.mdimx;

	std::cout << "N_directions = " << N_directions << std::endl;

	double cone_angle = 45.0; //(degrees)
	cone_angle = PI*cone_angle/180;

	trigProducts.initZeros(3, N_directions);

	Image<double> outputResolution;

	for (size_t dir=0; dir<N_directions; dir++)
	{
		outputResolution().initZeros(VRiesz);
//		MultidimArray<double> &pOutputResolution = outputResolution();
		double freq, freqL, freqH, counter, resolution_2;
		MultidimArray<int> mask_aux = mask();
		MultidimArray<int> &pMask = mask_aux;
		std::vector<double> list;
		double resolution;  //A huge value for achieving last_resolution < resolution

		double max_meanS = -1e38;
		double cut_value = 0.025;

		bool doNextIteration=true;

		int fourier_idx, last_fourier_idx = -1, iter = 0, fourier_idx_2;
		fourier_idx = aux_idx;
		int count_res = 0;
		double rot = MAT_ELEM(angles, 0, dir);
		double tilt = MAT_ELEM(angles, 1, dir);
		MAT_ELEM(trigProducts, 0, dir) = sin(tilt*PI/180)*cos(rot*PI/180);
		MAT_ELEM(trigProducts, 1, dir) = sin(tilt*PI/180)*sin(rot*PI/180);
		MAT_ELEM(trigProducts, 2, dir) = cos(tilt*PI/180);
		std::cout << "--------------NEW DIRECTION--------------" << std::endl;
		std::cout << "direction = " << dir+1 << "   rot = " << rot << "   tilt = " << tilt << std::endl;


		std::vector<float> noiseValues;
		FileName fnDebug;
		double last_resolution = 0;

		defineCone(fftV, conefilter, rot, tilt);
		maskMatrix.initConstant(1, NVoxelsOriginalMask, 1);
		do
		{
			continueIter = false;
			breakIter = false;
			//std::cout << "--------------Frequency--------------" << std::endl;

			resolution2eval_(fourier_idx, step,
							resolution, last_resolution, last_fourier_idx,
							freq, freqL, freqH,
							continueIter, breakIter, doNextIteration);

			if (breakIter)
				break;

			if (continueIter)
				continue;

			list.push_back(resolution);

			if (iter<2)
				resolution_2 = list[0];
			else
				resolution_2 = list[iter - 2];

			fnDebug = "Signal";

			amplitudeMonogenicSignal3D_fast(conefilter, freq, freqH, freqL, amplitudeMS, iter, dir, fnDebug, rot, tilt);

			double sumS=0, sumS2=0, sumN=0, sumN2=0, NN = 0, NS = 0;
			noiseValues.clear();


			double x_dir = sin(tilt*PI/180)*cos(rot*PI/180);
			double y_dir = sin(tilt*PI/180)*sin(rot*PI/180);
			double z_dir = cos(tilt*PI/180);

			double uz, uy, ux;

			int n=0;
			int z_size = ZSIZE(amplitudeMS);
			int x_size = XSIZE(amplitudeMS);
			int y_size = YSIZE(amplitudeMS);

			size_t idx_mask;
			idx_mask = 0;

			double amplitudeValue;

			for(int k=0; k<z_size; ++k)
			{
				for(int i=0; i<y_size; ++i)
				{
					for(int j=0; j<x_size; ++j)
					{
						if (DIRECT_MULTIDIM_ELEM(pMask, n)>=1)
						{
							if (MAT_ELEM(maskMatrix, 0, idx_mask) >0)
							{
							amplitudeValue=DIRECT_MULTIDIM_ELEM(amplitudeMS, n);
							sumS  += amplitudeValue;
							++NS;
							}
							++idx_mask;

						}
						else if (DIRECT_MULTIDIM_ELEM(pMask, n)==0)
						{
							uz = (k - z_size*0.5);
							ux = (j - x_size*0.5);
							uy = (i - y_size*0.5);

							double rad = sqrt(ux*ux + uy*uy + uz*uz);
							double iun = 1/rad;

							//BE CAREFULL with the order
							double dotproduct = (uy*y_dir + ux*x_dir + uz*z_dir)*iun;

							double acosine = acos(dotproduct);

							//TODO: change efficiency the if condition by means of fabs(cos(angle))
							if (((acosine<(cone_angle)) || (acosine>(PI-cone_angle)) )
									&& (rad>Rparticle))
							{
//								DIRECT_MULTIDIM_ELEM(coneVol, n) = 1;
								amplitudeValue=DIRECT_MULTIDIM_ELEM(amplitudeMS, n);
								noiseValues.push_back((float) amplitudeValue);
								sumN  += amplitudeValue;
								sumN2 += amplitudeValue*amplitudeValue;
								++NN;
							}
						}
						++n;
					}
				}
			}

			#ifdef DEBUG_DIR
				if (iter == 0)
				{
				Image<double> img;

				FileName iternumber;
				iternumber = formatString("cone_noise_%i_%i.vol", dir, iter);
				img = coneVol;
				img.write(iternumber);
				}
			#endif

			if ( (NS/(double) NVoxelsOriginalMask)<cut_value ) //when the 2.5% is reached then the iterative process stops
			{
				std::cout << "Search of resolutions stopped due to mask has been completed" << std::endl;
				doNextIteration =false;
				Nvoxels = 0;

				#ifdef DEBUG_MASK
				mask.write("partial_mask.vol");
				#endif
			}
			else
			{
				if (NS == 0)
				{
					std::cout << "There are no points to compute inside the mask" << std::endl;
					std::cout << "If the number of computed frequencies is low, perhaps the provided"
							"mask is not enough tight to the volume, in that case please try another mask" << std::endl;
					break;
				}

				double meanS=sumS/NS;
	//			double sigma2S=sumS2/NS-meanS*meanS;
				double meanN=sumN/NN;
				double sigma2N=sumN2/NN-meanN*meanN;

				if (meanS>max_meanS)
					max_meanS = meanS;

				if (meanS<0.001*AvgNoise)//0001*max_meanS)
				{
					//std::cout << "  meanS= " << meanS << " sigma2S= " << sigma2S << " NS	= " << NS << std::endl;
					//std::cout << "  meanN= " << meanN << " sigma2N= " << sigma2N << " NN= " << NN << std::endl;
					std::cout << "Search of resolutions stopped due to too low signal" << std::endl;
					std::cout << "\n"<< std::endl;
					doNextIteration = false;
				}
				else
				{
					// Check local resolution
					double thresholdNoise;
					//thresholdNoise = meanN+criticalZ*sqrt(sigma2N);

					std::sort(noiseValues.begin(),noiseValues.end());
					thresholdNoise = (double) noiseValues[size_t(noiseValues.size()*significance)];

					//std::cout << "thr="<< thresholdNoise << " " << meanN+criticalZ*sqrt(sigma2N) << " " << NN << std::endl;
					noiseValues.clear();

					std::cout << "Iteration = " << iter << ",   Resolution= " << resolution << ",   Signal = " << meanS << ",   Noise = " << meanN << ",  Threshold = " << thresholdNoise <<std::endl;


					size_t maskPos = 0;
					FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitudeMS)
					{
						if (DIRECT_MULTIDIM_ELEM(pMask, n)>=1)
						{
							if (MAT_ELEM(maskMatrix, 0, maskPos) >=1)
							{
								if (DIRECT_MULTIDIM_ELEM(amplitudeMS, n)>thresholdNoise)
								{
									MAT_ELEM(resolutionMatrix, dir, maskPos) = resolution;
									MAT_ELEM(maskMatrix, 0, maskPos) = 1;
								}
								else
								{
									MAT_ELEM(maskMatrix, 0, maskPos) += 1;
									if (MAT_ELEM(maskMatrix, 0, maskPos) >2)
									{
										MAT_ELEM(maskMatrix, 0, maskPos) = 0;
										MAT_ELEM(resolutionMatrix, dir, maskPos) = resolution_2;
									}
								}
							}
							++maskPos;
						}
					}

					if (doNextIteration)
						if (resolution <= (minRes-0.001))
							doNextIteration = false;
					}
			}
			++iter;
			last_resolution = resolution;
		}while(doNextIteration);

//		amplitudeMS.clear();
//		fftVRiesz.clear();

//		size_t maskPos=0;
//		Image<double> ResolutionVol;
//		MultidimArray<double> &pResolutionVol = ResolutionVol();
//
//		pResolutionVol.initZeros(amplitudeMS);
//		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(pResolutionVol)
//		{
//			if (DIRECT_MULTIDIM_ELEM(mask(), n) == 1)
//			{
//				double myres = MAT_ELEM(resolutionMatrix, dir, maskPos);
//				DIRECT_MULTIDIM_ELEM(pResolutionVol, n) = myres;
////				if (n == 14621798)
////					std::cout << maskPos << std::endl;
//				++maskPos;
//			}
//		}
//		#ifdef DEBUG_DIR
//		Image<double> saveImg;
//		saveImg = pResolutionVol;
//		FileName fnres = formatString("resolution_dir_%i.vol", dir+1);
//		saveImg.write(fnres);
//		saveImg.clear();
//		#endif
//		pResolutionVol.clear();
		list.clear();

		std::cout << "----------------direction-finished----------------" << std::endl;
	}




	////////////////////////////////////////////

	int maskPos = 0;

	//Remove outliers
	removeOutliers(trigProducts, resolutionMatrix);

	MultidimArray<double> radial, azimuthal, lowestResolution, highestResolution, doavol1, doavol2;
	MetaData prefDir;

	double radialThr, azimuthalThr;
	radialAzimuthalResolution(resolutionMatrix, mask(), radial, azimuthal,
			lowestResolution, highestResolution, doavol1, doavol2, radialThr, azimuthalThr, prefDir);

	Image<double> imgdoa;
	imgdoa = radial;
	imgdoa.write(fnradial);
	imgdoa = azimuthal;
	imgdoa.write(fnazimuthal);
	imgdoa = lowestResolution;
	imgdoa.write(fnLowestResolution);
	imgdoa = highestResolution;
	imgdoa.write(fnHighestResolution);
	imgdoa = doavol1;
	imgdoa.write(fnDoa1);
	imgdoa = doavol2;
	imgdoa.write(fnDoa2);

	MetaData mdRadialAzimuthalThr;
	size_t objIdx;
	objIdx = mdRadialAzimuthalThr.addObject();
	mdRadialAzimuthalThr.setValue(MDL_RESOLUTION_FREQ, radialThr, objIdx);
	mdRadialAzimuthalThr.setValue(MDL_RESOLUTION_FREQ2, azimuthalThr, objIdx);

	mdRadialAzimuthalThr.write(fnMDThr);

	std::cout << "radial = " << radialThr << "  azimuthal = " << azimuthalThr << std::endl;
	std::cout << "Calculating the radial and azimuthal resolution " << std::endl;


	MetaData mdRadial, mdAvg, mdHighest, mdLowest;

	Image<double> monores;
	monores.read(fnMonoRes);
	MultidimArray<double> monoresVol;
	monoresVol = monores();
	radialAverageInMask(mask(), radial, azimuthal, highestResolution, lowestResolution, monoresVol, mdAvg);

	mdAvg.write(fnRadialAvG);
}
