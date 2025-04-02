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

	rowprocess.getValue(MDL_SUBTRACTION_B, adjustParams.b); 
	rowprocess.getValue(MDL_SUBTRACTION_BETA0, adjustParams.b0); 
	rowprocess.getValue(MDL_SUBTRACTION_BETA1, adjustParams.b1); 
	
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

void ProgClassifyPartialOccupancy::logLikelihood(double ll_I, double ll_IsubP, const FileName &fnImgOut)
{	
	// Subtract ligand from particle
	IsubP() = (I() - adjustParams.b) - (P() * adjustParams.b0);

	// Por ahora solo consideramos ajuste de orden 0. 
	// Si queremos considerar el del orden 1 hay que que comprobar que b1 > 0
	// y ajustar por frecuencia
	
	// Apply adjustment order 0: PFourier0 = T(w) * PFourier = beta00 * PFourier
	// FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PFourier0)
	// {
	// 	int win = DIRECT_MULTIDIM_ELEM(wi, n);
	// 	if (win < maxwiIdx) 
	// 	{
	// 		DIRECT_MULTIDIM_ELEM(PFourier0,n) *= beta00;
	// 	}
	// }

	// Detect ligand regions
	binarizeMask(PmaskRoi);
	MultidimArray<double> PmaskRoiLabel;
	PmaskRoiLabel.resizeNoCopy(PmaskRoi());
	int numLig = labelImage2D(PmaskRoi(), PmaskRoiLabel, 8);

	#ifdef DEBUG_OUTPUT_FILES
	size_t dotPos = fnImgOut.find_last_of('.');
	Image<double> saveImage;

	saveImage = IsubP;
	saveImage.write(fnImgOut.substr(0, dotPos) + "_IsubP" + fnImgOut.substr(dotPos));

	saveImage = PmaskRoi;
	saveImage.write(fnImgOut.substr(0, dotPos) + "_PmaskRoiBinarize" + fnImgOut.substr(dotPos));

	saveImage() = PmaskRoiLabel;
	saveImage.write(fnImgOut.substr(0, dotPos) + "_PmaskRoiLabel" + fnImgOut.substr(dotPos));
	#endif

	// Calculate bounding box for each ligand region
	std::vector<int> minX(numLig, std::numeric_limits<int>::max());
	std::vector<int> minY(numLig, std::numeric_limits<int>::max());
	std::vector<int> maxX(numLig, 0);
	std::vector<int> maxY(numLig, 0);

	calculateBoundingBox(PmaskRoiLabel, minX, minY, maxX, maxY, numLig);

	// Calculate likelihood ofor each region
	MultidimArray<double> centeredLigand;
	MultidimArray<double> centeredLigandSubP;
	MultidimArray< std::complex<double> > fftI;
	MultidimArray< std::complex<double> > fftIsubP;

	// Analyze each ligand region independently
	for (size_t value = 0; value < numLig; ++value) 
	{
		#ifdef DEBUG_LOG_LIKELIHOOD
		std::cout << "Analyzign ligand region " << int(value +1) << std::endl;
		#endif
		// Cropping regions
		int width = maxX[value] - minX[value] + 1;
		int height = maxY[value] - minY[value] + 1;
		int centerX = Xdim / 2;
		int centerY = Ydim / 2;
		int numberOfPx = width * height;

		#ifdef DEBUG_LOG_LIKELIHOOD
		std::cout << "minX[value] " << minX[value] << std::endl;
		std::cout << "maxX[value] " << maxX[value] << std::endl;
		std::cout << "minY[value] " << minY[value] << std::endl;
		std::cout << "maxY[value] " << maxY[value] << std::endl;
		std::cout << "width " 		<< width << std::endl;
		std::cout << "height " 		<< height << std::endl;
		std::cout << "numberOfPx " 	<< numberOfPx << std::endl;
		#endif

		// Initialize new images for cropping
		centeredLigand.initZeros(Ydim, Xdim);
		centeredLigandSubP.initZeros(Ydim, Xdim);

		// Copy the region to the center of the new image
		for (int i = minY[value]; i <= maxY[value]; ++i) 
		{
			for (int j = minX[value]; j <= maxX[value]; ++j) 
			{
				int newI = centerY - height / 2 + (i - minY[value]);
				int newJ = centerX - width / 2 + (j - minX[value]);
				
				if (DIRECT_A2D_ELEM(PmaskRoiLabel, i, j) > 0)
				{
					DIRECT_A2D_ELEM(centeredLigand, newI, newJ) = DIRECT_A2D_ELEM(I(), i, j);
					DIRECT_A2D_ELEM(centeredLigandSubP, newI, newJ) = DIRECT_A2D_ELEM(IsubP(), i, j);
				}
			}
		}

		#ifdef DEBUG_OUTPUT_FILES
		size_t lastindex = fnImgOut.find_last_of(".");
		std::string debugFileFn = fnImgOut.substr(0, lastindex) + "_centeredLigand_" + std::to_string(value) + ".mrc";

		saveImage() = centeredLigand;
		saveImage.write(debugFileFn);
		#endif

		// Calculate FT for each cropping
		transformerI.FourierTransform(I(), fftI, false);
		transformerIsubP.FourierTransform(IsubP(), fftIsubP, false);

		// Calculate likelyhood for each region
		double ll_I_it = 0;
		double ll_IsubP_it = 0;

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftI)
		{		
			// Consider only those frequencies (under Nyquist) whose radial module is over threshold
			if (radialAvg_FT[DIRECT_MULTIDIM_ELEM(particleFreqMap,n)] > minModuleFT && DIRECT_MULTIDIM_ELEM(particleFreqMap,n) / Xdim <= 0.5)
			{
				ll_I_it     += (DIRECT_MULTIDIM_ELEM(fftI,n)     * std::conj(DIRECT_MULTIDIM_ELEM(fftI,n))).real()     / DIRECT_MULTIDIM_ELEM(powerNoise, n);
				ll_IsubP_it += (DIRECT_MULTIDIM_ELEM(fftIsubP,n) * std::conj(DIRECT_MULTIDIM_ELEM(fftIsubP,n))).real() / DIRECT_MULTIDIM_ELEM(powerNoise, n);
			}
		}

		// Normalize likelyhood by number of pixels of the crop adn take logarithms
		ll_I	 += std::log10(ll_I_it 	   / numberOfPx);
		ll_IsubP += std::log10(ll_IsubP_it / numberOfPx);

		#ifdef DEBUG_LOG_LIKELIHOOD
		std::cout << "ll_I_it for interation "     << value << " : " << ll_I_it     << ". Number of pixels: " << numberOfPx << std::endl;
		std::cout << "ll_IsubP_it for interation " << value << " : " << ll_IsubP_it << ". Number of pixels: " << numberOfPx << std::endl;
		#endif
	}

	std::cout << "ll_I: " << ll_I << "		ll_IsubP: " << ll_IsubP << std::endl;
}

void ProgClassifyPartialOccupancy::calculateBoundingBox(MultidimArray<double> PmaskRoiLabel, 
														std::vector<int> &minX, 
														std::vector<int> &minY, 
														std::vector<int> &maxX, 
														std::vector<int> &maxY, 
														int numLig)
{	
	for (size_t i = 0; i < Ydim; ++i) 
	{
		for (size_t j = 0; j < Xdim; ++j) 
		{
			int value = int(DIRECT_A2D_ELEM(PmaskRoiLabel, i, j));
			if (value != 0) 
			{
				if (j < minX[value - 1]) minX[value - 1] = j;
				if (j > maxX[value - 1]) maxX[value - 1] = j;
				if (i < minY[value - 1]) minY[value - 1] = i;
				if (i > maxY[value - 1]) maxY[value - 1] = i;
			}
		}
	}

	// Adjust bounding boxes to be squares
    for (int k = 0; k < numLig; ++k) 
	{
        int width = maxX[k] - minX[k] + 1;
        int height = maxY[k] - minY[k] + 1;
        int maxDim = std::max(width, height);

        int centerX = (minX[k] + maxX[k]) / 2;
        int centerY = (minY[k] + maxY[k]) / 2;

        minX[k] = centerX - maxDim / 2;
        maxX[k] = centerX + maxDim / 2;
        minY[k] = centerY - maxDim / 2;
        maxY[k] = centerY + maxDim / 2;

        // Ensure the bounding box stays within the image boundaries
        minX[k] = std::max(0, minX[k]);
        maxX[k] = std::min(static_cast<int>(Xdim - 1), maxX[k]);
        minY[k] = std::max(0, minY[k]);
        maxY[k] = std::min(static_cast<int>(Ydim - 1), maxY[k]);
    }
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
			std::cout << "-------Charactizing frequencies-------" << std::endl;
			frequencyCharacterization();
			std::cout << "-------Frequencies characterized-------" << std::endl;

			std::cout << "-------Estimating noise -------" << std::endl;
			noiseEstimation();
			std::cout << "-------Noise estimated -------" << std::endl;

		if (!realSpaceProjector)
		{
			// Initialize Fourier projectors
			std::cout << "-------Initializing projectors-------" << std::endl;

			projector = new FourierProjector(V(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);
			std::cout << "Volume ---> FourierProjector(V(),"<<padFourier<<","<<cutFreq<<","<<xmipp_transformation::BSPLINE3<<");"<< std::endl;

			std::cout << "-------Projectors initialized-------" << std::endl;
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

	logLikelihood(ll_I, ll_IsubP, fnImgOut);

	writeParticle(rowOut, fnImgOut, I, ll_I, ll_IsubP, (ll_I-ll_IsubP)); 
}


void ProgClassifyPartialOccupancy::noiseEstimation()
{
	auto t1 = std::chrono::high_resolution_clock::now();

	MetaData &mdIn = *getInputMd();

    srand(time(0)); // Seed for random number generation
    int maxX = Xdim - cropSize;
    int maxY = Ydim - cropSize;
    int minX = cropSize;
    int minY = cropSize;

	double scallignFactor = (Xdim * Ydim) / (cropSize * cropSize);

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

		// Optimize noise calulation: search for random regions that fall in the square that circunscribe the 
		// region that contain protein. We avoid generating random numbers in invalid regions.
		if(processedParticles < numberParticlesForBoundaryDetermination)
		{
			for (int i = 0; i < (int)Ydim; ++i) 
			{
				for (int j = 0; j < (int)Xdim; ++j) 
				{
					if (DIRECT_A2D_ELEM(PmaskProtein(), i, j) > 0) 
					{
						if (j < minX) minX = j;
						if (j > maxX) maxX = j;
						if (i < minY) minY = i;
						if (i > maxY) maxY = i;
					}
				}
			}
		}

		do {
			invalidRegion = false;
			noiseCrop.initZeros((int)Ydim, (int)Xdim);

			int x = minX + rand() % (maxX - minX + 1);
			int y = minY + rand() % (maxY - minY + 1);

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

					DIRECT_A2D_ELEM(noiseCrop,  (Ydim/2) - (cropSize/2) + i, (Xdim/2) - (cropSize/2) + j) = scallignFactor * DIRECT_A2D_ELEM(I(), y + i, x + j);
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

	powerNoise /= processedParticles;


	#ifdef DEBUG_OUTPUT_FILES
	Image<double> saveImage;
	size_t lastindex = fn_out.find_last_of(".");

	std::string debugFileFn = fn_out.substr(0, lastindex) + "_noisePower.mrc";

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


void ProgClassifyPartialOccupancy::frequencyCharacterization()
{
	// Calculate FT
	MultidimArray< std::complex<double> > V_ft; // Volume FT
    MultidimArray<double> V_ft_mod; // Volume FT modulus

	FourierTransformer fcTransformer;
	fcTransformer.FourierTransform(V(), V_ft, false);

	// FT dimensions
	int Xdim_ft = XSIZE(V_ft);
	int Ydim_ft = YSIZE(V_ft);
	int Zdim_ft = ZSIZE(V_ft);
	int Ndim_ft = NSIZE(V_ft);

	if (Zdim_ft == 1)
	{
		Zdim_ft = Ndim_ft;
	}

	int maxRadius = std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft));	// Restric analysis to Nyquist

	#ifdef DEBUG_FREQUENCY_PROFILE
	std::cout << "FFT map dimensions: " << std::endl;  
	std::cout << "FT xSize " << Xdim_ft << std::endl;
	std::cout << "FT ySize " << Ydim_ft << std::endl;
	std::cout << "FT zSize " << Zdim_ft << std::endl;
	std::cout << "FT nSize " << Ndim_ft << std::endl;
	std::cout << "maxRadius " << maxRadius << std::endl;
	#endif

	// Construct frequency map and initialize the frequency vectors
	MultidimArray<double> freqMap;
	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;
	Matrix1D<double> freq_fourier_z;

	freq_fourier_x.initZeros(Xdim_ft);
	freq_fourier_y.initZeros(Ydim_ft);
	freq_fourier_z.initZeros(Zdim_ft);

	double u;	// u is the frequency

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities
	VEC_ELEM(freq_fourier_z,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Zdim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Zdim, u);
		VEC_ELEM(freq_fourier_z, k) = u;
	}

	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Ydim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Ydim, u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Xdim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Xdim, u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}

	//Initializing map with frequencies
	freqMap.resizeNoCopy(V_ft);

	// Directional frequencies along each direction
	double uz;
	double uy;
	double ux;
	double uz2;
	double uz2y2;
	long n=0;
	int idx = 0;

	for(size_t k=0; k<Zdim_ft; ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz2 = uz*uz;
		
		for(size_t i=0; i<Ydim_ft; ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uz2y2 = uz2 + uy*uy;

			for(size_t j=0; j<Xdim_ft; ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				ux = sqrt(uz2y2 + ux*ux);

				idx = (int) round(ux * Xdim);
				DIRECT_MULTIDIM_ELEM(freqMap,n) = idx;

				++n;
			}
		}
	}

	// Calculate radial average
	std::vector<double> radialCounter_FT;

	V_ft_mod.initZeros(Zdim_ft, Ydim_ft, Xdim_ft);
	radialAvg_FT.resize(maxRadius, 0);
	radialCounter_FT.resize(maxRadius, 0);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V_ft)
	{
		double value_mod  = sqrt((DIRECT_MULTIDIM_ELEM(V_ft,n) * std::conj(DIRECT_MULTIDIM_ELEM(V_ft,n))).real());

		DIRECT_MULTIDIM_ELEM(V_ft_mod,n)  = value_mod;
		
		if(DIRECT_MULTIDIM_ELEM(freqMap,n) < maxRadius)
		{
			radialAvg_FT[(int)(DIRECT_MULTIDIM_ELEM(freqMap,n))]  += value_mod;
			radialCounter_FT[(int)(DIRECT_MULTIDIM_ELEM(freqMap,n))] += 1;
		}
	}

	for(size_t i = 0; i < radialCounter_FT.size(); i++)
	{
		radialAvg_FT[i] /= radialCounter_FT[i];
	}

	// Calculate minimum modulus for frequency 
	auto maxElement = std::max_element(radialAvg_FT.begin(), radialAvg_FT.end());
	minModuleFT = 0.25*static_cast<double>(*maxElement);

	//Construct particle frequency map (2D)
	// Reuse freq_fourier_x and freq_fourier_y vectors

	//Initializing particle map with frequencies
	particleFreqMap.initZeros(Ydim_ft, Xdim_ft);

	// Directional frequencies along each direction
	double uy2;
	n=0;
	idx = 0;

	for(size_t i=0; i<Ydim_ft; ++i)
	{
		uy = VEC_ELEM(freq_fourier_y, i);
		uy2 = uy*uy;

		for(size_t j=0; j<Xdim_ft; ++j)
		{
			ux = VEC_ELEM(freq_fourier_x, j);
			ux = sqrt(uy2 + ux*ux);

			idx = (int) round(ux * Xdim);
			DIRECT_MULTIDIM_ELEM(particleFreqMap,n) = idx;

			++n;
		}
	}

	#ifdef DEBUG_OUTPUT_FILES
	// Save FT maps
	size_t lastindex = fn_out.find_last_of(".");
	std::string rawname = fn_out.substr(0, lastindex);
	Image<double> saveImage;
	
	std::string debugFileFn = rawname + "_freqMap.mrc";
	saveImage() = freqMap;
	saveImage.write(debugFileFn);

	debugFileFn = rawname + "_particleFreqMap.mrc";
	saveImage() = particleFreqMap;
	saveImage.write(debugFileFn);

	debugFileFn = rawname + "_FT_mod.mrc";
	saveImage() = V_ft_mod;
	saveImage.write(debugFileFn);

	// Save output metadata
	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < radialCounter_FT.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_AVG, radialAvg_FT[i],      id);
		md.setValue(MDL_X,   radialCounter_FT[i],  id);
	}

	std::string outputMD = rawname + "_FT_profile.xmd";
	md.write(outputMD);
	#endif

	std::cout << "Frenquency profiling estimated. Minimum modulus value: " << minModuleFT << std::endl;
}
