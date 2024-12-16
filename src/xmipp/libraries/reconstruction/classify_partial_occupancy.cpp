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


 // Empty constructor =======================================================
ProgClassifyPartialOccupancy::ProgClassifyPartialOccupancy()
{
	produces_a_metadata = true;
    each_image_produces_an_output = true;
    keep_input_columns = true;
	save_metadata_stack = true;
    remove_disabled = false;
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
	sigma=getIntParam("--sigma");
	sampling = getDoubleParam("--sampling");
	padFourier = getDoubleParam("--padding");
    maxResol = getDoubleParam("--max_resolution");

	if (checkParam("--cirmaskrad"))
	{
		cirmaskrad = getDoubleParam("--cirmaskrad");
		maskVolProvided = false;
	}
	else if (checkParam("--mask"))
	{
		fnMaskVol = getParam("--mask");
		maskVolProvided = true;
	}
	
	fnProj = getParam("--save"); 
	nonNegative = checkParam("--nonNegative");
	boost = checkParam("--boost");
	subtract = checkParam("--subtract");
	realSpaceProjector = checkParam("--realSpaceProjection");
 }

 // Show ====================================================================
 void ProgClassifyPartialOccupancy::show() const
 {
    if (!verbose)
        return;
	std::cout
	<< "Input particles:\t" << fnParticles << std::endl
	<< "Reference volume:\t" << fnVolR << std::endl
	<< "Mask of the region of interest to keep or subtract:\t" << fnMaskRoi << std::endl
	<< "Sigma of low pass filter:\t" << sigma << std::endl
	<< "Sampling rate:\t" << sampling << std::endl
	<< "Padding factor:\t" << padFourier << std::endl
    << "Max. Resolution:\t" << maxResol << std::endl
	<< "Output particles:\t" << fnOut << std::endl;
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
    addParamsLine("[--mask_roi <mask_roi=\"\">]     : 3D mask for region of interest to keep or subtract, no mask implies subtraction of whole images");
	addParamsLine("[--sampling <sampling=1>]		: Sampling rate (A/pixel)");
	addParamsLine("[--max_resolution <f=-1>]		: Maximum resolution in Angtroms up to which the substraction is calculated. \
													  By default (-1) it is set to sampling/sqrt(2).");
	addParamsLine("[--padding <p=2>]				: Padding factor for Fourier projector");
	addParamsLine("[--sigma <s=1>]					: Decay of the filter (sigma) to smooth the mask transition");
	addParamsLine("[--nonNegative]					: Ignore particles with negative beta0 or R2");
	addParamsLine("[--boost]						: Perform a boosting of original particles");
	addParamsLine("[--save <structure=\"\">]		: Path for saving intermediate files");
	addParamsLine("[--subtract]						: The mask contains the region to SUBTRACT");
	addParamsLine("[--realSpaceProjection]			: Project volume in real space to avoid Fourier artifacts");

	// Example
    addExampleLine("A typical use is:",false);
    addExampleLine("xmipp_subtract_projection -i input_particles.xmd --ref input_map.mrc --mask_roi mask_vol.mrc "
				   "-o output_particles --sampling 1 --max_resolution 4");
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
	rowOut.setValue(MDL_SUBTRACTION_R2, R2a); 
	rowOut.setValue(MDL_SUBTRACTION_BETA0, b0save); 
	rowOut.setValue(MDL_SUBTRACTION_BETA1, b1save); 
	rowOut.setValue(MDL_SUBTRACTION_B, b); 
	rowOut.setValue(MDL_AVG, avg); 
	rowOut.setValue(MDL_STDDEV, std); 
	rowOut.setValue(MDL_ZSCORE, zScore); 
	if (nonNegative && (disable || R2a < 0))
	{
		rowOut.setValue(MDL_ENABLED, -1);
	}
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
	
	// Project volume + apply translation, CTF and mask
	// If provided, mask already have been applied to volume
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
	const auto sizeI = (int)XSIZE(I());
	MultidimArray<double> &mI=I();

	#ifdef DEBUG_OUTPUT_FILES
	MultidimArray<double> maskedIdiff;
	maskedIdiff.initZeros(mIdiff);
	#endif

	double sum = 0;
	double sum2 = 0;
	int Nelems = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(M())
	{
		if (DIRECT_MULTIDIM_ELEM(M(),n) > 0)
		{
			double value = DIRECT_MULTIDIM_ELEM(mIdiff, n);

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
			double value = DIRECT_MULTIDIM_ELEM(mIdiff, n);

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

	Image<double> saveImage;
	saveImage() = maskedIdiff;
	saveImage.write(fnMaskedImgOut);

	M.write(fnImgOut.substr(0, dotPos) + "_mask" + fnImgOut.substr(dotPos));
	#endif
}

 // Main methods ===================================================================
void ProgClassifyPartialOccupancy::preProcess() 
{
	// Read input volume, mask and particles metadata
	show();
	V.read(fnVolR);
	V().setXmippOrigin();

	// Create 2D circular mask to avoid edge artifacts after wrapping
	size_t Xdim;
	size_t Ydim;
	size_t Zdim;
	size_t Ndim;
	V.getDimensions(Xdim, Ydim, Zdim, Ndim);

	// Read or create mask keep and compute inverse of mask keep (mask subtract)
	vM.read(fnM);
	vM().setXmippOrigin();

	double cutFreq = 0.5 * (sampling/maxResol);

	if (rank==0)
	{
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
	// Initialize aux variable
	disable = false;

	// Project volume and process projections 
	const auto sizeI = (int)XSIZE(I());
	processParticle(rowIn, sizeI);

	// Build projected and final masks. Mask projection is always calculated in real space
	projectVolume(vM(), PmaskRoi, sizeI, sizeI, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);

	#ifdef DEBUG_OUTPUT_FILES
	size_t dotPos = fnImgOut.find_last_of('.');
	PmaskRoi.write(fnImgOut.substr(0, dotPos) + "_PmaskRoi" + fnImgOut.substr(dotPos));
	#endif

	// Apply binarization, shift and gaussian filter to the projected mask
	M = binarizeMask(PmaskRoi);

	#ifdef DEBUG_OUTPUT_FILES
	size_t dotPos = fnImgOut.find_last_of('.');
	M.write(fnImgOut.substr(0, dotPos) + "_M" + fnImgOut.substr(dotPos));
	#endif

	// Create empty new image for output particle
	MultidimArray<double> &mIw=Iw();
	mIdiff.initZeros(I());
	mIdiff.setXmippOrigin();

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mIw)
		DIRECT_MULTIDIM_ELEM(mIw,n) = (DIRECT_MULTIDIM_ELEM(I(),n) * DIRECT_MULTIDIM_ELEM(P(),n));
	
	#ifdef DEBUG_OUTPUT_FILES
	dotPos = fnImgOut.find_last_of('.');
	P.write(fnImgOut.substr(0, dotPos) + "_P" + fnImgOut.substr(dotPos));
	I.write(fnImgOut.substr(0, dotPos) + "_I" + fnImgOut.substr(dotPos));
	Iw.write(fnImgOut.substr(0, dotPos) + "_Iw" + fnImgOut.substr(dotPos));
	#endif

	// Compute particle stats after subtraction
	double avg;
	double std;
	double zScore;

	computeParticleStats(mIw, M, fnImgOut, avg, std, zScore);

	writeParticle(rowOut, fnImgOut, mIw, avg, std, zScore); 
}

void ProgClassifyPartialOccupancy::postProcess()
{
	getOutputMd().write(fn_out);
}
