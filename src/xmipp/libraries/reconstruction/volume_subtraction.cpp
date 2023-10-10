/***************************************************************************
 *
 * Authors:    Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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

#include "volume_subtraction.h"
#include "core/transformations.h"
#include <core/histogram.h>
#include <core/xmipp_fftw.h>
#include <core/xmipp_program.h>
#include <data/fourier_filter.h>
#include <core/geometry.h>


 // Empty constructor =======================================================
ProgVolumeSubtraction::ProgVolumeSubtraction()
{
	produces_a_metadata = true;
    each_image_produces_an_output = true;
    keep_input_columns = true;
	save_metadata_stack = true;
    remove_disabled = false;
	rank = 0;
}

// Usage ===================================================================
void ProgVolumeSubtraction::defineParams() 
{
	// Usage
	addUsageLine("This program modifies a volume as much as possible in order to assimilate it to another one,");
	addUsageLine("without loosing the important information in it process. Then, the subtraction of the two volumes");
	addUsageLine("can be optionally calculated. Sharpening: reference volume must be an atomic structure previously");
	addUsageLine("converted into a density map of the same specimen than in input volume 2.");
	// Parameters
	XmippMetadataProgram::defineParams();
	addParamsLine("--ref <volume>\t: Reference volume");
	addParamsLine("[--sub]\t: Perform the subtraction of the volumes. Output will be the difference");
	addParamsLine("[--sigma <s=3>]\t: Decay of the filter  (sigma) to smooth the mask transition");
	addParamsLine("[--iter <n=5>]\t: Number of iterations for the adjustment process");
	addParamsLine("[--mask1 <mask=\"\">]\t: Mask for volume 1");
	addParamsLine("[--mask2 <mask=\"\">]\t: Mask for volume 2");
	addParamsLine("[--maskSub <mask=\"\">]\t: Mask for subtraction region");
	addParamsLine("[--cutFreq <f=0>]\t: Filter both volumes with a filter which specified cutoff frequency "
			"(i.e. resolution inverse, <0.5)");
	addParamsLine("[--lambda <l=1>]\t: Relaxation factor for Fourier Amplitude POCS, i.e. 'how much modification "
			"of volume Fourier amplitudes', between 1 (full modification, recommended) and 0 (no modification)");
	addParamsLine("[--radavg]\t: Match the radially averaged Fourier amplitudes when adjusting the amplitudes instead "
			"of taking directly them from the reference volume");
	addParamsLine("[--computeEnergy]\t: Compute the energy difference between each step (energy difference gives "
			"information about the convergence of the adjustment process, while it can slightly slow the performance)");
	addParamsLine("[--saveV1 <structure=\"\"> ]\t: Save subtraction intermediate file (vol1 filtered) just when option "
			"--sub is passed, if not passed the input reference volume is not modified");
	addParamsLine("[--saveV2 <structure=\"\"> ]\t: Save subtraction intermediate file (vol2 adjusted) just when option "
			"--sub is passed, if not passed the output of the program is this file");
	addParamsLine("[--subtomos]\t: indicate that input is a set of subtomograms");
}

// Read arguments ==========================================================
void ProgVolumeSubtraction::readParams() 
{
	XmippMetadataProgram::readParams();
	fnVolRef = getParam("--ref");
	performSubtraction = checkParam("--sub");
	iter = getIntParam("--iter");
	sigma = getIntParam("--sigma");
	fnMask1 = getParam("--mask1");
	fnMask2 = getParam("--mask2");
	fnMaskSub = getParam("--maskSub");
	cutFreq = getDoubleParam("--cutFreq");
	lambda = getDoubleParam("--lambda");
	fnVol1F = getParam("--saveV1");
	if (fnVol1F.isEmpty())
		fnVol1F = "volume1_filtered.mrc";
	fnVol2A = getParam("--saveV2");
	if (fnVol2A.isEmpty())
		fnVol2A = "volume2_adjusted.mrc";
	radavg = checkParam("--radavg");
	computeE = checkParam("--computeEnergy");
	subtomos = 	performSubtraction = checkParam("--subtomos");

}

// Show ====================================================================
void ProgVolumeSubtraction::show() const {
	if (!verbose)
        return;
	std::cout 
	<< "Input volume(s):\t" << fnVolMd << std::endl
	<< "Reference volume:\t" << fnVolRef << std::endl
	<< "Input mask 1:\t" << fnMask1 << std::endl
	<< "Input mask 2:\t" << fnMask2 << std::endl
	<< "Input mask subtract:\t" << fnMaskSub << std::endl
	<< "Sigma:\t" << sigma << std::endl
	<< "Iterations:\t" << iter << std::endl
	<< "Cutoff frequency:\t" << cutFreq << std::endl
	<< "Relaxation factor:\t" << lambda << std::endl
	<< "Match radial averages:\t" << radavg << std::endl
	<< "Output:\t" << fnOut << std::endl;
}

void ProgVolumeSubtraction::readParticle(const MDRow &r) {
	r.getValueOrDefault(MDL_IMAGE, fnVol2, "no_filename");
	V.read(fnVol2);
	V().setXmippOrigin();
	V.write("V.mrc");
 }

 void ProgVolumeSubtraction::writeParticle(MDRow &rowOut, FileName fnImgOut, Image<double> &img) {
	img.write(fnImgOut);
	rowOut.setValue(MDL_IMAGE, fnImgOut);
 }

/* Methods used to adjust an input volume (V) to a another reference volume (V1) through
the use of Projectors Onto Convex Sets (POCS) */
void ProgVolumeSubtraction::POCSmask(const MultidimArray<double> &mask, MultidimArray<double> &I) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
		DIRECT_MULTIDIM_ELEM(I, n) *= DIRECT_MULTIDIM_ELEM(mask, n);
}

void ProgVolumeSubtraction::POCSnonnegative(MultidimArray<double> &I) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
		DIRECT_MULTIDIM_ELEM(I, n) = std::max(0.0, DIRECT_MULTIDIM_ELEM(I, n));
}

void ProgVolumeSubtraction::POCSFourierAmplitude(const MultidimArray<double> &V1FourierMag,
		MultidimArray<std::complex<double>> &V2Fourier, double l) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V1FourierMag) {
		double mod = std::abs(DIRECT_MULTIDIM_ELEM(V2Fourier, n));
		if (mod > 1e-10) // Condition to avoid divide by zero, values smaller than
			// this threshold are considered zero
			DIRECT_MULTIDIM_ELEM(V2Fourier, n) *=
					((1 - l) + l * DIRECT_MULTIDIM_ELEM(V1FourierMag, n)) / mod;
	}
}

void ProgVolumeSubtraction::POCSFourierAmplitudeRadAvg(MultidimArray<std::complex<double>> &V,
		double l, const MultidimArray<double> &rQ, int V1size_x, int V1size_y, int V1size_z) {
	int V1size2_x = V1size_x/2;
	double V1sizei_x = 1.0/V1size_x;
	int V1size2_y = V1size_y/2;
	double V1sizei_y = 1.0/V1size_y;
	int V1size2_z = V1size_z/2;
	double V1sizei_z = 1.0/V1size_z;
	double wx;
	double wy;
	double wz;
	for (int k=0; k<ZSIZE(V); ++k)
	{
		FFT_IDX2DIGFREQ_FAST(k,V1size_z,V1size2_z,V1sizei_z,wz)
		double wz2 = wz*wz;
		for (int i=0; i<YSIZE(V); ++i)
		{
			FFT_IDX2DIGFREQ_FAST(i,V1size_y,V1size2_y,V1sizei_y,wy)
			double wy2_wz2 = wy*wy + wz2;
			for (int j=0; j<XSIZE(V); ++j)
			{
				FFT_IDX2DIGFREQ_FAST(j,V1size_x,V1size2_x,V1sizei_x,wx)
				double w = sqrt(wx*wx + wy2_wz2);
				auto iw = std::min((int)floor(w*V1size_x), (int)XSIZE(rQ) -1);
				DIRECT_A3D_ELEM(V,k,i,j)*=(1-l)+l*DIRECT_MULTIDIM_ELEM(rQ,iw);
			}
		}
	}
}

void ProgVolumeSubtraction::POCSMinMax(MultidimArray<double> &V, double v1m, double v1M) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V) {
		double val = DIRECT_MULTIDIM_ELEM(V, n);
		if (val < v1m)
			DIRECT_MULTIDIM_ELEM(V, n) = v1m;
		else if (val > v1M)
			DIRECT_MULTIDIM_ELEM(V, n) = v1M;
	}
}

void ProgVolumeSubtraction::POCSFourierPhase(const MultidimArray<std::complex<double>> &phase,
		MultidimArray<std::complex<double>> &FI) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(phase)
    						DIRECT_MULTIDIM_ELEM(FI, n) =
    								std::abs(DIRECT_MULTIDIM_ELEM(FI, n)) * DIRECT_MULTIDIM_ELEM(phase, n);
}

/* Other methods needed to pre-process and operate with the volumes */
void ProgVolumeSubtraction::extractPhase(MultidimArray<std::complex<double>> &FI) const{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FI) {
		auto *ptr = (double *)&DIRECT_MULTIDIM_ELEM(FI, n);
		double phi = atan2(*(ptr + 1), *ptr);
		DIRECT_MULTIDIM_ELEM(FI, n) = std::complex<double>(cos(phi), sin(phi));
	}
}

void ProgVolumeSubtraction::computeEnergy(MultidimArray<double> &Vdif, const MultidimArray<double> &Vact) const{
	Vdif = Vdif - Vact;
	double energy = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Vdif)
	energy += DIRECT_MULTIDIM_ELEM(Vdif, n) * DIRECT_MULTIDIM_ELEM(Vdif, n);
	energy = sqrt(energy / MULTIDIM_SIZE(Vdif));
}

void ProgVolumeSubtraction::centerFFTMagnitude(MultidimArray<double> &VolRad,
		MultidimArray<std::complex<double>> &VolFourierRad,
		MultidimArray<double> &VolFourierMagRad) const{
	FourierTransformer transformerRad;
	transformerRad.completeFourierTransform(VolRad, VolFourierRad);
	CenterFFT(VolFourierRad, true);
	FFT_magnitude(VolFourierRad, VolFourierMagRad);
	VolFourierMagRad.setXmippOrigin();
}

void ProgVolumeSubtraction::radialAverage(const MultidimArray<double> &VolFourierMag,
		const MultidimArray<double> &V, MultidimArray<double> &radial_mean) {
	MultidimArray<double> radial_count;
	int Vsize2_x = int(XSIZE(V))/2;
	double Vsizei_x = 1.0/int(XSIZE(V));
	int Vsize2_y = int(YSIZE(V))/2;
	double Vsizei_y = 1.0/int(YSIZE(V));
	int Vsize2_z = int(ZSIZE(V))/2;
	double Vsizei_z = 1.0/int(ZSIZE(V));
	double wx;
	double wy;
	double wz;
	auto maxrad = int(floor(sqrt(Vsize2_x*Vsize2_x + Vsize2_y*Vsize2_y + Vsize2_z*Vsize2_z)));
	radial_count.initZeros(maxrad);
	radial_mean.initZeros(maxrad);
	for (int k=0; k<Vsize2_z; ++k)
		{
			FFT_IDX2DIGFREQ_FAST(k,ZSIZE(V),Vsize2_z,Vsizei_z,wz)
			double wz2 = wz*wz;
			for (int i=0; i<Vsize2_y; ++i)
			{
				FFT_IDX2DIGFREQ_FAST(i,YSIZE(V),Vsize2_y,Vsizei_y,wy)
				double wy2_wz2 = wy*wy + wz2;
				for (int j=0; j<Vsize2_x; ++j)
				{
					FFT_IDX2DIGFREQ_FAST(j,XSIZE(V),Vsize2_x,Vsizei_x,wx)
					double w = sqrt(wx*wx + wy2_wz2);
					auto iw = (int)round(w*int(XSIZE(V)));
					DIRECT_A1D_ELEM(radial_mean,iw)+=DIRECT_A3D_ELEM(VolFourierMag,k,i,j);
					DIRECT_A1D_ELEM(radial_count,iw)+=1.0;
				}
			}
		}
	radial_mean/= radial_count;
}

MultidimArray<double> ProgVolumeSubtraction::computeRadQuotient(const MultidimArray<double> &v1Mag,
		const MultidimArray<double> &vMag, const MultidimArray<double> &V1,
		const MultidimArray<double> &V) {
	// Compute the quotient of the radial mean of the volumes to use it in POCS amplitude
	MultidimArray<double> radial_meanV1;
	radialAverage(v1Mag, V1, radial_meanV1);
	std::cout << "---41---" << radial_meanV1 << std::endl;
	MultidimArray<double> radial_meanV;
	radialAverage(vMag, V, radial_meanV);
	std::cout << "---42---" << radial_meanV << std::endl;
	MultidimArray<double> radQuotient = radial_meanV1 / radial_meanV;
	std::cout << "---43---" << radQuotient << std::endl;
	FOR_ALL_ELEMENTS_IN_ARRAY1D(radQuotient)
		radQuotient(i) = std::min(radQuotient(i), 1.0);
	std::cout << "---44---" << std::endl;
	return radQuotient;
}

void ProgVolumeSubtraction::createFilter(FourierFilter &filter2, double cutFreq) {
	filter2.FilterBand = LOWPASS;
	filter2.FilterShape = RAISED_COSINE;
	filter2.raised_w = 0.02;
	filter2.w1 = cutFreq;
}

Image<double> ProgVolumeSubtraction::subtraction(Image<double> V1, Image<double> &V,
		const MultidimArray<double> &mask, const FileName &fnVol1F,
		const FileName &fnVol2A, FourierFilter &filter2, double cutFreq) {
	Image<double> V1Filtered;
	V1Filtered() = V1();
	if (cutFreq != 0){
		filter2.applyMaskSpace(V1Filtered());
	}
	if (!fnVol1F.isEmpty()) {
		V1Filtered.write(fnVol1F);
	}
	if (!fnVol2A.isEmpty()) {
		V.write(fnVol2A);
	}
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V1())
    						DIRECT_MULTIDIM_ELEM(V1(), n) =
    								DIRECT_MULTIDIM_ELEM(V1(), n) * (1 - DIRECT_MULTIDIM_ELEM(mask, n)) +
									(DIRECT_MULTIDIM_ELEM(V1Filtered(), n) -
											std::min(DIRECT_MULTIDIM_ELEM(V(), n),
													DIRECT_MULTIDIM_ELEM(V1Filtered(), n))) *
													DIRECT_MULTIDIM_ELEM(mask, n);
	return V1;
}

MultidimArray<double> ProgVolumeSubtraction::computeMagnitude(MultidimArray<double> &volume) {
	FourierTransformer transformer;
	MultidimArray<std::complex<double>> fourier;
	MultidimArray<double> magnitude;
	transformer.FourierTransform(volume, fourier, false);
	FFT_magnitude(fourier, magnitude);
	return magnitude;
}

MultidimArray<double> ProgVolumeSubtraction::createMask(const Image<double> &volume, const FileName &fnM1, const FileName &fnM2) {
	MultidimArray<double> mask;
	if (fnM1 != "" && fnM2 != "") {
		Image<double> mask1;
		Image<double> mask2;
		mask1.read(fnM1);
		std::cout << "---mask1---" << std::endl;
		mask1.write("mask1.mrc");
		mask2.read(fnM2);
		std::cout << "---mask2---" << std::endl;
		mask2.write("mask2.mrc");
		mask = mask1() * mask2();
	} else {
		mask.resizeNoCopy(volume());
		mask.initConstant(1.0);
	}
	return mask;
}

void ProgVolumeSubtraction::filterMask(MultidimArray<double> &mask) const{
	FourierFilter Filter;
	Filter.FilterShape = REALGAUSSIAN;
	Filter.FilterBand = LOWPASS;
	Filter.w1 = sigma;
	Filter.applyMaskSpace(mask);
}

MultidimArray<std::complex<double>> ProgVolumeSubtraction::computePhase(MultidimArray<double> &volume) {
	MultidimArray<std::complex<double>> phase;
	transformer2.FourierTransform(volume, phase, true);
	extractPhase(phase);
	return phase;
}

MultidimArray<double> ProgVolumeSubtraction::getSubtractionMask(const FileName &fnMSub, MultidimArray<double> mask){
	if (fnMSub.isEmpty()){
		filterMask(mask);
		return mask;
	}
	else {
		Image<double> masksub;
		masksub.read(fnMSub);
		return masksub();
	}
}

void ProgVolumeSubtraction::preProcess() {
	// Read and preprocess reference volume (V1)
	show();
	/*Image<double> V1;
	V1.read(fnVolRef);
	V1.write("V1.mrc");
	mask = createMask(V1, fnMask1, fnMask2);
	POCSmask(mask, V1());
	POCSnonnegative(V1());
	V1().computeDoubleMinMax(v1min, v1max);
	std1 = V1().computeStddev();
	createFilter(filter2, cutFreq);*/

 }

/* Core of the program: processing needed to adjust input volume V2 to reference volume V1. 
Several iteration of this processing should be run. */

void ProgVolumeSubtraction::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut) {
	Image<double> V1;
	V1.read(fnVolRef);
	V1().setXmippOrigin();
	mask = createMask(V1, fnMask1, fnMask2);
	POCSmask(mask, V1());
	POCSnonnegative(V1());
	V1().computeDoubleMinMax(v1min, v1max);
	std1 = V1().computeStddev();
	createFilter(filter2, cutFreq);
	
	readParticle(rowIn);
	MultidimArray<double> &mv = V();
	mv.setXmippOrigin();

	Image<double> Vf;
	Vf = V;
	MultidimArray<double> &mvf = Vf();
	mvf.setXmippOrigin();
	mvf.initZeros();

	if (subtomos)
	{
		// Window subtomo (padding) before apply alignment
		MultidimArray <double> &mpad = padv();
		mpad.setXmippOrigin();
		pad = 2 * XSIZE(mv);
		mv.window(mpad,STARTINGY(mv)*(int)pad, STARTINGX(mv)*(int)pad, FINISHINGY(mv)*(int)pad, FINISHINGX(mv)*(int)pad);
		// Read alignment
		rowIn.getValueOrDefault(MDL_ANGLE_ROT, part_angles.rot, 0);
		rowIn.getValueOrDefault(MDL_ANGLE_TILT, part_angles.tilt, 0);
		rowIn.getValueOrDefault(MDL_ANGLE_PSI, part_angles.psi, 0);
		roffset.initZeros(3);
		rowIn.getValueOrDefault(MDL_SHIFT_X, roffset(0), 0);
		rowIn.getValueOrDefault(MDL_SHIFT_Y, roffset(1), 0);
		rowIn.getValueOrDefault(MDL_SHIFT_Z, roffset(2), 0);
		roffset *= -1;
		// Apply alignment
		Euler_rotate(mv, part_angles.rot, part_angles.tilt, part_angles.psi, mvf);
		selfTranslate(xmipp_transformation::LINEAR, mvf, roffset, xmipp_transformation::WRAP);

		//Crop to restore original size
		mpad.window(mvf,STARTINGY(mvf), STARTINGX(mvf), FINISHINGY(mvf), FINISHINGX(mvf));
	}

	POCSmask(mask, mvf);
	POCSnonnegative(mvf);
	Vdiff = Vf;
	auto V2FourierPhase = computePhase(mvf);
	auto V1FourierMag = computeMagnitude(V1());
	auto V2FourierMag = computeMagnitude(mvf);
	auto radQuotient = computeRadQuotient(V1FourierMag, V2FourierMag, V1(), mvf);
	std::cout << "---5---" << std::endl;
	for (n = 0; n < iter; ++n) 
	{
		transformer2.FourierTransform(mvf, V2Fourier, false);
		std::cout << "---6---" << std::endl;
		if (radavg) {
			auto V1size_x = (int)XSIZE(V1());
			auto V1size_y = (int)YSIZE(V1());
			auto V1size_z = (int)ZSIZE(V1());
			std::cout << "---7---" << std::endl;
			POCSFourierAmplitudeRadAvg(V2Fourier, lambda, radQuotient, V1size_x, V1size_y, V1size_z);
			std::cout << "---8---" << std::endl;
		}
		else {
			std::cout << "---9---" << std::endl;
			POCSFourierAmplitude(V1FourierMag, V2Fourier, lambda);
			std::cout << "---10---" << std::endl;
		}
		std::cout << "---11---" << std::endl;
		transformer2.inverseFourierTransform();
		if (computeE) {
			std::cout << "---12---" << std::endl;
			computeEnergy(Vdiff(), mvf);
			Vdiff = Vf;
		}
		std::cout << "---13---" << std::endl;
		POCSMinMax(mvf, v1min, v1max);
		std::cout << "---14---" << std::endl;
		if (computeE) {
			computeEnergy(Vdiff(), mvf);
			std::cout << "---15---" << std::endl;
			Vdiff = Vf;
		}
		POCSmask(mask, mvf);
		std::cout << "---16---" << std::endl;
		if (computeE) {
			computeEnergy(Vdiff(), mvf);
			std::cout << "---17---" << std::endl;
			Vdiff = Vf;
		}
		transformer2.FourierTransform();
		std::cout << "---18---" << std::endl;
		POCSFourierPhase(V2FourierPhase, V2Fourier);
		std::cout << "---19---" << std::endl;
		transformer2.inverseFourierTransform();
		if (computeE) {
			computeEnergy(Vdiff(), mvf);
			Vdiff = Vf;
		}
		POCSnonnegative(mvf);
		std::cout << "---20---" << std::endl;
		if (computeE) {
			computeEnergy(Vdiff(), mvf);
			Vdiff = Vf;
		}
		double std2 = mvf.computeStddev();
		std::cout << "---21---" << std::endl;
		mvf *= std1 / std2;
		if (computeE) {
			computeEnergy(Vdiff(), mvf);
			Vdiff = Vf;
		}
		std::cout << "---22---" << std::endl;
		if (cutFreq != 0) {
			filter2.generateMask(mvf);
			std::cout << "---23---" << std::endl;
			filter2.do_generate_3dmask = true;
			filter2.applyMaskSpace(mvf);
			std::cout << "---24---" << std::endl;
			if (computeE) {
				computeEnergy(Vdiff(), mvf);
				Vdiff = Vf;
			}
		}
		std::cout << "---25---" << std::endl;
	}
	std::cout << "---26---" << std::endl;

	if (performSubtraction) {
		std::cout << "---27---" << std::endl;
		auto masksub = getSubtractionMask(fnMaskSub, mask);
		std::cout << "---28---" << std::endl;
		V1.read(fnVolRef);
		std::cout << "---29---" << std::endl;
		V = subtraction(V1, Vf, masksub, fnVol1F, fnVol2A, filter2, cutFreq);
		std::cout << "---30---" << std::endl;
	}

	if (subtomos)
	{
		// Recover original alignment
		Euler_rotate(mvf, -part_angles.rot, -part_angles.tilt, -part_angles.psi, mvf);
		std::cout << "---31---" << std::endl;
		selfTranslate(xmipp_transformation::LINEAR, mvf, -roffset, xmipp_transformation::WRAP);
		std::cout << "---32---" << std::endl;
	}
	std::cout << "---33---" << std::endl;
	writeParticle(rowOut, fnImgOut, Vf); 

}

void ProgVolumeSubtraction::postProcess()
{
	std::cout << "---34---" << std::endl;
	getOutputMd().write(fn_out);
}
