/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gomez (federico.pdeisidro@astx.com)
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

 #include "classify_partial_occupancy.h"
 #include "core/transformations.h"
 #include "core/metadata_extension.h"
 #include "core/multidim_array.h"
 #include "core/xmipp_image_extension.h"
 #include "core/xmipp_image_generic.h"
 #include "core/xmipp_image_base.h"
 #include "core/xmipp_fft.h"
 #include "core/xmipp_fftw.h"
 #include "core/linear_system_helper.h"
 #include "data/projection.h"
 #include "data/mask.h"
 #include "data/filters.h"
 #include "data/morphology.h"
 #include <iostream>
 #include <string>
 #include <sstream>
 #include "data/image_operate.h"
 #include <iostream>
 #include <cstdlib>
 #include <vector>
 #include <utility>
 #include <chrono>


 // Empty constructor =======================================================
ProgClassifyPartialOccupancy::ProgClassifyPartialOccupancy()
{
	produces_a_metadata = true;
    each_image_produces_an_output = true;
    keep_input_columns = true;
	save_metadata_stack = true;
	projector = nullptr;
	rank = 0;
}

ProgClassifyPartialOccupancy::~ProgClassifyPartialOccupancy()
{
	delete projector;
}

 // Read arguments ==========================================================
 void ProgClassifyPartialOccupancy::readParams()
 {
	XmippMetadataProgram::readParams();
 	fnVolR = getParam("--ref");
	fnMaskRoi=getParam("--mask_roi");
	fnMaskProtein=getParam("--mask_protein");
	padFourier = getDoubleParam("--padding");
	realSpaceProjector = checkParam("--realSpaceProjection");
 }

 // Show ====================================================================
 void ProgClassifyPartialOccupancy::show() const
 {
    if (!verbose)
        return;
	std::cout
	<< "Input particles:\t" << fn_in << std::endl
	<< "Reference volume:\t" << fnVolR << std::endl
	<< "Mask of the protein region:\t" << fnMaskProtein << std::endl
	<< "Mask of the region of interest to keep or subtract:\t" << fnMaskRoi << std::endl
	<< "Padding factor:\t" << padFourier << std::endl
	<< "Output particles:\t" << fn_out << std::endl;
 }

 // Usage ===================================================================
 void ProgClassifyPartialOccupancy::defineParams()
 {
	//Usage
    addUsageLine("This program computes the subtraction between particles and a reference"); 
	addUsageLine(" volume, by computing its projections with the same angles that input particles have."); 
	addUsageLine(" Then, each particle and the correspondent projection of the reference volume are numerically");
	addUsageLine(" adjusted and subtracted using a mask which denotes the region of interest to keep or subtract.");

    //Parameters
	XmippMetadataProgram::defineParams();
    addParamsLine("--ref <volume>\t: Reference volume to subtract");
    addParamsLine("--mask_protein <mask_roi=\"\">	: 3D mask for region of the specimen");
    addParamsLine("--mask_roi <mask_roi=\"\">     	: 3D mask for region of interest to keep or subtract, no mask implies subtraction of whole images");
	addParamsLine("[--realSpaceProjection]			: Project volume in real space to avoid Fourier artifacts");
	addParamsLine("[--padding <p=2>]				: Padding factor for Fourier projector");

	// Example
    addExampleLine("A typical use is:",false);
    addExampleLine("xmipp_subtract_projection -i input_particles.xmd --ref input_map.mrc --mask_roi mask_roi_vol.mrc --mask_protein mask_protein_vol.mrc -o output_particles");
 }

 // I/O methods ===================================================================
 void ProgClassifyPartialOccupancy::readParticle(const MDRow &r) 
 {
	r.getValueOrDefault(MDL_IMAGE, fnImgI, "no_filename");
	I.read(fnImgI);
	I().setXmippOrigin();
 }

 void ProgClassifyPartialOccupancy::writeParticle(MDRow &rowOut, FileName fnImgOut, Image<double> &img, double avg, double std, double zScore) 
 {
	img.write(fnImgOut);

	rowOut.setValue(MDL_IMAGE, fnImgOut);
	rowOut.setValue(MDL_AVG, avg); 
	rowOut.setValue(MDL_STDDEV, std); 
	rowOut.setValue(MDL_ZSCORE, zScore); 
 }

 // Utils methods ===================================================================
 Image<double> ProgClassifyPartialOccupancy::binarizeMask(Projection &m) const 
 {
	MultidimArray<double> &mm=m();

 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mm)
		DIRECT_MULTIDIM_ELEM(mm,n) = (DIRECT_MULTIDIM_ELEM(mm,n)>0) ? 1:0; 
 	return m;
 }

 Image<double> ProgClassifyPartialOccupancy::invertMask(const Image<double> &m) 
 {
	PmaskI = m;
	MultidimArray<double> &mPmaskI=PmaskI();
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mPmaskI)
		DIRECT_MULTIDIM_ELEM(mPmaskI,n) = (DIRECT_MULTIDIM_ELEM(mPmaskI,n)*(-1))+1;
	return PmaskI;
 }

void ProgClassifyPartialOccupancy::processParticle(const MDRow &rowprocess, int sizeImg) 
{	
	// Read metadata information for projection
	readParticle(rowprocess);
	rowprocess.getValueOrDefault(MDL_ANGLE_ROT, part_angles.rot, 0);
	rowprocess.getValueOrDefault(MDL_ANGLE_TILT, part_angles.tilt, 0);
	rowprocess.getValueOrDefault(MDL_ANGLE_PSI, part_angles.psi, 0);
	roffset.initZeros(2);
	rowprocess.getValueOrDefault(MDL_SHIFT_X, roffset(0), 0);
	rowprocess.getValueOrDefault(MDL_SHIFT_Y, roffset(1), 0);
	roffset *= -1;
	
	// Project volume + apply translation
	if (realSpaceProjector)
	{
		projectVolume(V(), P, sizeImg, sizeImg, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);
	}
	else
	{
		projectVolume(*projector, P, sizeImg, sizeImg, part_angles.rot, part_angles.tilt, part_angles.psi, ctfImage);
		selfTranslate(xmipp_transformation::LINEAR, P(), roffset, xmipp_transformation::WRAP);
	}
}

void ProgClassifyPartialOccupancy::computeParticleStats(Image<double> &I, Image<double> &M, FileName fnImgOut, double &avg, double &std, double &zScore)
{	
	MultidimArray<double> &mI=I();

	double sum = 0;
	double sum2 = 0;
	int Nelems = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(M())
	{
		if (DIRECT_MULTIDIM_ELEM(M(),n) > 0)
		{
			double value = DIRECT_MULTIDIM_ELEM(mI, n);

			sum += value;
			sum2 += value*value;
			++Nelems;
		}
	}

	avg = sum / Nelems;
	std = sqrt(sum2/Nelems - avg*avg);
	int zScoreThr = 3;
	zScore = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(M())
	{
		if (DIRECT_MULTIDIM_ELEM(M(),n) > 0)
		{
			double value = DIRECT_MULTIDIM_ELEM(mI, n);

			zScore += (value - avg) / std;
			// if(value > (avg + std * zScoreThr))
			// {
			// 	zScore++;
			// }
		}
	}

	zScore /= Nelems;
	
	#ifdef DEBUG
	std::cout << "sum " << sum << std::endl;
	std::cout << "sum2 " << sum2 << std::endl;
	std::cout << "Nelems " << Nelems << std::endl;
	std::cout << "avg " << avg << std::endl;
	std::cout << "std " << std << std::endl;
	std::cout << "zScore " << zScore << std::endl;
	#endif

	#ifdef DEBUG_OUTPUT_FILES
	// Save output masked particle for debugging
	FileName fnMaskedImgOut;
	size_t dotPos = fnImgOut.find_last_of('.');

    if (dotPos == std::string::npos) {
        // No extension found
        fnMaskedImgOut = fnImgOut + "_masked";
    }
    fnMaskedImgOut = fnImgOut.substr(0, dotPos) + "_masked" + fnImgOut.substr(dotPos);

	M.write(fnImgOut.substr(0, dotPos) + "_mask" + fnImgOut.substr(dotPos));
	#endif
}

void ProgClassifyPartialOccupancy::logLikelyhood(double ll_I, double ll_IsubP)
{	
	MultidimArray< std::complex<double> > fftI;
	transformerI.FourierTransform(I(), fftI, false);

	IsubP() = I() - P();
	MultidimArray< std::complex<double> > fftIsubP;
	transformerIsubP.FourierTransform(IsubP(), fftIsubP, false);

	std::cout << "DIRECT_MULTIDIM_ELEM(powerNoise, 0): " << DIRECT_MULTIDIM_ELEM(powerNoise, 0) << std::endl;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftI)
	{
		ll_I     += (DIRECT_MULTIDIM_ELEM(fftI,n)     * std::conj(DIRECT_MULTIDIM_ELEM(fftI,n))).real()     / (1 + DIRECT_MULTIDIM_ELEM(powerNoise, n));
		ll_IsubP += (DIRECT_MULTIDIM_ELEM(fftIsubP,n) * std::conj(DIRECT_MULTIDIM_ELEM(fftIsubP,n))).real() / (1 + DIRECT_MULTIDIM_ELEM(powerNoise, n));

	}

	std::cout << "ll_I: " << ll_I << "		ll_IsubP: " << ll_IsubP << std::endl;
}

 // Main methods ===================================================================
void ProgClassifyPartialOccupancy::preProcess() 
{
	// Read input volume, mask and particles metadata
	show();
	V.read(fnVolR);
	V().setXmippOrigin();

	// Read Protein mask
	vMaskP.read(fnMaskProtein);
	vMaskP().setXmippOrigin();

	// Read ROI mask
	vMaskRoi.read(fnMaskRoi);
	vMaskRoi().setXmippOrigin();

	// Create 2D circular mask to avoid edge artifacts after wrapping
	V.getDimensions(Xdim, Ydim, Zdim, Ndim);

	I().initZeros((int)Ydim, (int)Xdim);  //*** dimensions should be read from particles

	// Initialize projectors
	double cutFreq = 0.5;

	if (rank==0)
	{
		if (!realSpaceProjector)
		{
			// Initialize Fourier projectors
			std::cout << "-------Initializing projectors-------" << std::endl;

			projector = new FourierProjector(V(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);
			std::cout << "Volume ---> FourierProjector(V(),"<<padFourier<<","<<cutFreq<<","<<xmipp_transformation::BSPLINE3<<");"<< std::endl;

			std::cout << "-------Projectors initialized-------" << std::endl;

			std::cout << "-------Estimating noise -------" << std::endl;

			noiseEstimation();

			std::cout << "-------Noise estimated -------" << std::endl;
		}
	}
	else
	{
		if (!realSpaceProjector)
		{
			projector = new FourierProjector(padFourier, cutFreq, xmipp_transformation::BSPLINE3);
		}
	}
 }

void ProgClassifyPartialOccupancy::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
 { 
	// Project volume and process projections 
	const auto sizeI = (int)XSIZE(I());

	processParticle(rowIn, sizeI);

	// Build projected and final masks. Mask projection is always calculated in real space
	projectVolume(vMaskP(), PmaskProtein, sizeI, sizeI, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);
	projectVolume(vMaskRoi(), PmaskRoi, sizeI, sizeI, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);

	// Apply binarization to projected mask, DO NOT NEEDED BECAUSE PROJECTING IN REAL SPACE
	// M_P = binarizeMask(PmaskProtein);
	// M = binarizeMask(PmaskRoi);

	#ifdef DEBUG_OUTPUT_FILES
	size_t dotPos = fnImgOut.find_last_of('.');
	I.write(fnImgOut.substr(0, dotPos) + "_I" + fnImgOut.substr(dotPos));
	P.write(fnImgOut.substr(0, dotPos) + "_P" + fnImgOut.substr(dotPos));
	PmaskProtein.write(fnImgOut.substr(0, dotPos) + "_PmaskProtein" + fnImgOut.substr(dotPos));
	PmaskRoi.write(fnImgOut.substr(0, dotPos) + "_PmaskRoi" + fnImgOut.substr(dotPos));
	// M_P.write(fnImgOut.substr(0, dotPos) + "_PmaskProtein_Norm" + fnImgOut.substr(dotPos));
	// M.write(fnImgOut.substr(0, dotPos) + "_PmaskRoi_Norm" + fnImgOut.substr(dotPos));
	#endif

	double ll_I = 0;
	double ll_IsubP = 0;

	logLikelyhood(ll_I, ll_IsubP);

	writeParticle(rowOut, fnImgOut, I, ll_I, ll_IsubP, 0); 
}


void ProgClassifyPartialOccupancy::noiseEstimation()
{
	auto t1 = std::chrono::high_resolution_clock::now();

	MetaData &mdIn = *getInputMd();

    srand(time(0)); // Seed for random number generation
    int maxX = Xdim - cropSize;
    int maxY = Ydim - cropSize;

    bool invalidRegion;
	size_t processedParticles = 0;

    MultidimArray< double > noiseCrop;
	powerNoise.initZeros((int)Ydim, (int)Xdim/2 +1);

	// Iterate particles
	for (const auto& r : mdIn)
	{
		#ifdef DEBUG_NOISE_CALCULATION
		std::cout << "Estimating noise from particle " << processedParticles + 1 << std::endl;
		#endif

		r.getValueOrDefault(MDL_IMAGE, fnImgI, "no_filename");
		I.read(fnImgI);
		I().setXmippOrigin();

		r.getValueOrDefault(MDL_ANGLE_ROT, part_angles.rot, 0);
		r.getValueOrDefault(MDL_ANGLE_TILT, part_angles.tilt, 0);
		r.getValueOrDefault(MDL_ANGLE_PSI, part_angles.psi, 0);
		roffset.initZeros(2);
		r.getValueOrDefault(MDL_SHIFT_X, roffset(0), 0);
		r.getValueOrDefault(MDL_SHIFT_Y, roffset(1), 0);
		roffset *= -1;

		projectVolume(vMaskP(), PmaskProtein, Xdim, Ydim, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);
		projectVolume(vMaskRoi(), PmaskRoi, Xdim, Ydim, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);

		do {
			invalidRegion = false;
			noiseCrop.initZeros((int)Ydim, (int)Xdim);

			int x = rand() % maxX;
			int y = rand() % maxY;

			for (size_t i = 0; i < cropSize; i++)
			{
				for (size_t j = 0; j < cropSize; j++)
				{
					#ifdef DEBUG_NOISE_CALCULATION
					std::cout << "--------------------------------------------------" << std::endl;
					std::cout << "x  " << x << " y " << y  << std::endl;
					std::cout << "i " << i << " j  " << j  << std::endl;
					std::cout << "y + i  " << y + i << " x + j " << x + j << std::endl;
					std::cout << "(Ydim/2) " << (Ydim/2) << " (Xdim/2) " << (Xdim/2) << std::endl;
					std::cout << "(Ydim/2) - (cropSize/2) + i  " << (Ydim/2) - (cropSize/2) + i << " (Xdim/2) - (cropSize/2) + j " << (Xdim/2) - (cropSize/2) + j << std::endl;
					#endif

					if (DIRECT_A2D_ELEM(PmaskProtein(), y + i, x + j) == 0 || DIRECT_A2D_ELEM(PmaskRoi(), y + i, x + j) > 0)
					{
						invalidRegion = true;

						#ifdef DEBUG_NOISE_CALCULATION
					 	std::cout << "Invalid region. Trying again..." << std::endl;
						#endif
					
						break;
					}

					DIRECT_A2D_ELEM(noiseCrop,  (Ydim/2) - (cropSize/2) + i, (Xdim/2) - (cropSize/2) + j) = DIRECT_A2D_ELEM(I(), y + i, x + j);
				}

				if (invalidRegion) {
					break;
				}
		}
		} while (invalidRegion);

		#ifdef DEBUG_NOISE_CALCULATION
		size_t lastindex = fn_out.find_last_of(".");
		std::string rawname = fn_out.substr(0, lastindex);

		Image<double> saveImage;
		std::string debugFileFn = rawname + "_noiseCrop.mrc";

		saveImage() = noiseCrop;
		saveImage.write(debugFileFn);
		#endif

	    FourierTransformer transformerNoise;
		transformerNoise.FourierTransform(noiseCrop, noiseSpectrum, false);

		#ifdef DEBUG_NOISE_CALCULATION
		MultidimArray< double > noiseSpectrumReal;
		noiseSpectrumReal.initZeros(YSIZE(noiseSpectrum), XSIZE(noiseSpectrum)); 

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(noiseSpectrum)
			DIRECT_MULTIDIM_ELEM(noiseSpectrumReal,n) = DIRECT_MULTIDIM_ELEM(noiseSpectrum,n).real();

		Image<double> saveImageHalf;
		debugFileFn = rawname + "_noiseSpectrumReal.mrc";

		saveImageHalf() = noiseSpectrumReal;
		saveImageHalf.write(debugFileFn);
		#endif

 		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(noiseSpectrum)
			DIRECT_MULTIDIM_ELEM(powerNoise,n) += (DIRECT_MULTIDIM_ELEM(noiseSpectrum,n) * std::conj(DIRECT_MULTIDIM_ELEM(noiseSpectrum,n))).real();
			
		#ifdef DEBUG_NOISE_CALCULATION
		std::cout << "Noise estimated from particle " << processedParticles + 1 << " sucessfully." << std::endl;
		#endif

		processedParticles++;

		if (processedParticles == numberParticlesForNoiseEstimation)
		{
			break;
		}
	}

	// powerNoise /= processedParticles;


	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fn_out.find_last_of(".");
	std::string rawname = fn_out.substr(0, lastindex);

	Image<double> saveImage;
	std::string debugFileFn = rawname + "_noisePower.mrc";

	saveImage() = powerNoise;
	saveImage.write(debugFileFn);
	#endif

	#ifdef VERBOSE_OUTPUT
	auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1);	// Getting number of seconds as an integer
	std::cout << "Execution time for noise estimation: " << ms_int.count() << " seconds." << std::endl;

	std::cout << "Number of particles processed for noise estimation: " << processedParticles << std::endl;
	#endif

}
