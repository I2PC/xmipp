/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
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

#include "tomo_align_subtomo_stacks.h"
#include <core/bilib/kernel.h>
#include <core/metadata_extension.h>
#include <core/metadata_vec.h>
#include <core/matrix2d.h>
#include <core/geometry.h>
#include <numeric>

void ProgTomoAlignSubtomoStacks::readParams()
{
	fnIn = getParam("-i");
	fnIniVol = getParam("--initVol");
	fnOut = getParam("-o");
	fnGal = getParam("--gallery");
	maxFreq = getDoubleParam("--maxFreq");
	nbest = getIntParam("--nbest");
	nthrs = getIntParam("--threads");
}


void ProgTomoAlignSubtomoStacks::defineParams()
{
	addUsageLine("This function takes a tomogram an extract a set of subtomogram from it. The coordinates of the subtomograms are speciffied in the metadata given by coordinates.");
	addParamsLine("  -i <xmd_file=\"\">                 : Metadata (.xmd file) with the subtomograms stacks");
	addParamsLine("  --initVol <mrc_file=\"\">          : Initial Volume");
	addParamsLine("  -o <mrc_file=\"\">                 : path to the output directory. ");
	addParamsLine("  --maxFreq <maxFreq=0.5>            : Maximum digital frequency.");
	addParamsLine("  --gallery <xmd_file=\"\">          : path to the gallery of projections.");
	addParamsLine("  [--nbest <nbest=3>]                : Number of candidates to be considered as true alignment");
	addParamsLine("  [--threads <s=4>]                  : Number of threads");
}


void ProgTomoAlignSubtomoStacks::alignAgainstGallery(MultidimArray<double> &img0, MetaDataVec &mdGallery)
{
	FileName fn;

	Image<double> gallery;
	auto &imgGal = gallery();
	double corrValue = 0;
	double rot, tilt, psi;
	std::vector<double> vecRot;
	std::vector<double> vecTilt;
	std::vector<double> vecPsi;

	// bestValues is a matrix with 4 columns (corrValue, rot, tilt psi) and as rows the number of best elements
	Matrix2D<double> bestValues;
	bestValues.initZeros(nbest, 2);

	size_t idx = 0;
	for (const auto& row : mdGallery)
	{
		row.getValue(MDL_IMAGE, fn);
		row.getValue(MDL_ANGLE_ROT, rot);
		row.getValue(MDL_ANGLE_TILT, tilt);
		row.getValue(MDL_ANGLE_PSI, psi);

		vecRot.push_back(rot);
		vecTilt.push_back(tilt);
		vecPsi.push_back(psi);

		gallery.read(fn);

		//TODO: Fourier transform
		corr2(img0, imgGal, corrValue);

		if (corrValue > MAT_ELEM(bestValues, 0, 0))
		{
			for (size_t i = 1; i<nbest-1; i++)
			{
				if (corrValue < MAT_ELEM(bestValues, i, 0))
				{
					MAT_ELEM(bestValues, i-1, 0) = corrValue;
					MAT_ELEM(bestValues, i-1, 1) = idx;
				}
				else
				{
					MAT_ELEM(bestValues, i-1, 0) = MAT_ELEM(bestValues, i, 0);
					MAT_ELEM(bestValues, i-1, 1) = MAT_ELEM(bestValues, i, 1);
				}
			}
			if (MAT_ELEM(bestValues, nbest-1, 0) < corrValue)
			{
				MAT_ELEM(bestValues, nbest-1, 0) = corrValue;
				MAT_ELEM(bestValues, nbest-1, 0) = idx;
			}
		}
		idx++;
	}
}

void ProgTomoAlignSubtomoStacks::freqMask(const MultidimArray< std::complex<double> > &mapfftV, const MultidimArray<double> &inputVol, double &maxDigFreq)
{
	MultidimArray< double > freqElems;
	size_t xvoldim;
	size_t yvoldim;
	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;
	// Initializing the frequency vectors
	freq_fourier_y.initZeros(YSIZE(mapfftV));
	freq_fourier_x.initZeros(XSIZE(mapfftV));

	// u is the frequency
	double u;

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities
	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<YSIZE(mapfftV); ++k){
		FFT_IDX2DIGFREQ(k,YSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<XSIZE(mapfftV); ++k){
		FFT_IDX2DIGFREQ(k,XSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}

	// Directional frequencies along each direction
	double uy, ux, uz2y2;
	long n=0;
	int idx = 0;

	for(size_t i=0; i<YSIZE(mapfftV); ++i)
	{
		uy = VEC_ELEM(freq_fourier_y, i);
		uz2y2 = uy*uy;

		for(size_t j=0; j<XSIZE(mapfftV); ++j)
		{
			ux = VEC_ELEM(freq_fourier_x, j);
			ux = sqrt(uz2y2 + ux*ux);
			if	(ux<=maxDigFreq)
			{
				freqIdx.push_back(n);
			}
			++n;
		}
	}
}

void ProgTomoAlignSubtomoStacks::fourierDistance(MultidimArray<double> &img0, MultidimArray<double> &imgGal, double &dist2)
{
	MultidimArray<std::complex<double>> FTimgGal, FTimg0;
	FourierTransformer transformer1(FFTW_BACKWARD);
	transformer1.FourierTransform(imgGal, FTimgGal);//, false);
	FourierTransformer transformer2(FFTW_BACKWARD);
	FTimg0.resizeNoCopy(FTimgGal);
	transformer2.FourierTransform(img0, FTimg0);

	double dist = 0;
	for (size_t i=0; i<freqIdx.size(); ++i)
	{
		auto n = freqIdx[i];
		dist = abs(DIRECT_MULTIDIM_ELEM(FTimgGal, n)-DIRECT_MULTIDIM_ELEM(FTimg0, n));
		dist2 += dist*dist;
	}
}



void ProgTomoAlignSubtomoStacks::corr2(MultidimArray<double> &img0, MultidimArray<double> &imgGal, double &corrVal)
{
	double sumX = 0;
	double sumY = 0;
	double N = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img0)
	{
		sumX += DIRECT_MULTIDIM_ELEM(img0, n);
		sumY += DIRECT_MULTIDIM_ELEM(imgGal, n);
		N += 1.0;
	}

	double meanX = sumX/N;
	double meanY = sumY/N;


	double sumValTot = 0;

	double num = 0;
	double den1 = 0;
	double den2 = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img0)
	{
		double xVol = DIRECT_MULTIDIM_ELEM(img0, n) - meanX;
		double yVol = DIRECT_MULTIDIM_ELEM(imgGal, n) - meanY;

		num += (xVol) * (yVol);
		den1 += (xVol) * (xVol);
		den2 += (yVol) * (yVol);
	}

	corrVal = std::max(0.0, num/sqrt(den1*den2));
}

/*
void ProgTomoAlignSubtomoStacks::listofParticles(MetaDataVec &md, std::vector<size_t> &listparticlesIds)
{
	size_t parId;
	std::vector<size_t> targetparId;
	bool addParticleId = false;
	for (const auto& row : md)
	{
		row.getValue(MDL_PARTICLE_ID, parId);
		row.getValue(MDL_ANGLE_TILT, tilt);

		for (size_t i = 0; i<targetparId.size(); i++)
		{
			if (targetparId[i] != parId)
			{

			}
		}
	}
}
*/

void ProgTomoAlignSubtomoStacks::validateAlignment(MultidimArray<double> &initVol, MultidimArray<double> &imgExp, double &rot, double &tilt, double &psi, int ydim, int xdim)
{
	Projection imgPrj;
	MultidimArray<double> &ptrImg = imgPrj();
	projectVolume(projector, imgPrj, ydim, xdim, rot, tilt, psi);

	double corrVal;
	corr2(ptrImg, imgExp, corrVal);
}

template <typename T>
std::vector<size_t> ProgTomoAlignSubtomoStacks::sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
	std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

void ProgTomoAlignSubtomoStacks::sortAndRemoveMdParticles(MetaDataVec &mdParticles)
{
	FileName fn;
	size_t partId;

	std::vector<size_t> listOfParticlesId;
	std::vector<FileName> listOfFn;

	// Getting filenames and values
	for (const auto& row : mdParticles)
	{
		row.getValue(MDL_IMAGE, fn);
		row.getValue(MDL_PARTICLE_ID, partId);

		listOfFn.push_back(fn);
		listOfParticlesId.push_back(partId);
	}

	// removing isolating particles
	std::vector<size_t> sortedIds;
	sortedIds = sort_indexes(listOfParticlesId);

	size_t lastId  = SIZE_MAX;
	int countCandidates = 1;
	for (size_t i=0; i<sortedIds.size(); i++)
	{
		auto idx = sortedIds[i];
		auto pId = listOfParticlesId[idx];

		if (pId != lastId)
		{
			lastId = pId;
			if (countCandidates ==1)
			{
				continue;
			}
			countCandidates = 1;
		}

	}
}

void ProgTomoAlignSubtomoStacks::run()
{
	std::cout << "Starting ... "<< std::endl;

	// Reading initial volume and prepare projector
	Image<double> initVolImg;
	initVolImg.read(fnIniVol);
	auto &initVol = initVolImg();

	int xdim = XSIZE(initVol);
	int ydim = YSIZE(initVol);

	int BSplinedegree = 3;
	projector = FourierProjector(initVol, 2.0, maxFreq, BSplinedegree);
	Projection imgPrj;



	FileName fn;
	std::vector<FileName> fnStack;
	std::vector<double> tiltStack;
	std::string tsId;
	size_t parId;

	Matrix2D<double> eulerMat;
	double rot, tilt, psi;

	Image<double> image0;
	auto &img0 = image0();


	// This is only use for full alignment
	/*
	MetaDataVec md, mdGallery;
	md.read(fnIn);
	mdGallery.read(fnGal);
	*/
	MetaDataVec md, mdGallery;
	md.read(fnIn);
	mdGallery.read(fnGal);

//	for (const auto& row : md)
//	{
//
//	}

	/*
	for (const auto& row : md)
	{
		row.getValue(MDL_IMAGE, fn);
		row.getValue(MDL_TSID, tsId);
		row.getValue(MDL_PARTICLE_ID, parId);
		row.getValue(MDL_ANGLE_TILT, tilt);

		//
		//if (abs(tilt)<1)
		//{

		//}

		image0.read(fn);

		alignAgainstGallery(img0, mdGallery);

		eulerMat.initZeros(4, 4);

		Euler_matrix2angles(eulerMat, rot, tilt, psi, true);
		projectVolume(projector, imgPrj, ydim, xdim, rot, tilt, psi);
	}
	*/

}

