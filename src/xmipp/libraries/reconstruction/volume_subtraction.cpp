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


// Usage ===================================================================
void ProgVolumeSubtraction::defineParams() {
	// Usage
	addUsageLine("This program modifies a volume as much as possible in order "
			"to assimilate it to another one, "
			"without loosing the important information in it ('adjustment "
			"process'). Then, the subtraction of "
			"the two volumes can be optionally calculated. Sharpening: "
			"reference volume must be an atomic "
			"structure previously converted into a density map of the "
			"same specimen than in input volume 2.");
	// Parameters
	addParamsLine("--i1 <volume>			: Reference volume");
	addParamsLine("--i2 <volume>			: Volume to modify");
	addParamsLine("[-o <structure=\"\">]\t: Volume 2 modified or "
			"volume difference");
	addParamsLine("\t: If no name is given, "
			"then output_volume.mrc");
	addParamsLine("[--sub]\t: Perform the "
			"subtraction of the volumes. Output will be the difference");
	addParamsLine("[--sigma <s=3>]\t: Decay of the filter "
			"(sigma) to smooth the mask transition");
	addParamsLine(
			"[--iter <n=5>]\t: Number of iterations for the adjustment process");
	addParamsLine("[--mask1 <mask=\"\">]		: Mask for volume 1");
	addParamsLine("[--mask2 <mask=\"\">]		: Mask for volume 2");
	addParamsLine(
			"[--maskSub <mask=\"\">]\t: Mask for subtraction region");
	addParamsLine(
			"[--cutFreq <f=0>]\t: Filter both volumes with a filter which "
			"specified cutoff frequency (i.e. resolution inverse, <0.5)");
	addParamsLine(
			"[--lambda <l=1>]\t: Relaxation factor for Fourier Amplitude POCS, "
			"i.e. 'how much modification of volume Fourier amplitudes', between 1 "
			"(full modification, recommended) and 0 (no modification)");
	addParamsLine("[--radavg]\t: Match the radially averaged Fourier "
			"amplitudes when adjusting the amplitudes instead of taking "
			"directly them from the reference volume");
	addParamsLine(
			"[--computeEnergy]\t: Compute the energy difference between each step "
			"(energy difference gives information about the convergence of the "
			"adjustment process, while it can slightly slow the performance)");
	addParamsLine(
			"[--saveV1 <structure=\"\"> ]\t: Save subtraction intermediate file "
			"(vol1 filtered) just when option --sub is passed, if not passed the "
			"input reference volume is not modified");
	addParamsLine(
			"[--saveV2 <structure=\"\"> ]\t: Save subtraction intermediate file "
			"(vol2 adjusted) just when option --sub is passed, if not passed the "
			"output of the program is this file");
}

// Read arguments ==========================================================
void ProgVolumeSubtraction::readParams() {
	fnVol1 = getParam("--i1");
	fnVol2 = getParam("--i2");
	fnOutVol = getParam("-o");
	if (fnOutVol.isEmpty())
		fnOutVol = "output_volume.mrc";
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
}

// Show ====================================================================
void ProgVolumeSubtraction::show() const {
	std::cout << "Input volume 1:\t" << fnVol1 << std::endl
			<< "Input volume 2:    	   	" << fnVol2 << std::endl
			<< "Input mask 1:    	   	" << fnMask1 << std::endl
			<< "Input mask 2:    	   	" << fnMask2 << std::endl
			<< "Input mask sub:		" << fnMaskSub << std::endl
			<< "Sigma:			" << sigma << std::endl
			<< "Iterations:			" << iter << std::endl
			<< "Cutoff frequency:		" << cutFreq << std::endl
			<< "Relaxation factor:		" << lambda << std::endl
			<< "Match radial averages:\t" << radavg << std::endl
			<< "Output:\t" << fnOutVol << std::endl;
}

/* Methods used to adjust an input volume (V) to a another reference volume (V1) through
the use of Projectors Onto Convex Sets (POCS) */
void POCSmask(const MultidimArray<double> &mask, MultidimArray<double> &I) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
		DIRECT_MULTIDIM_ELEM(I, n) *= DIRECT_MULTIDIM_ELEM(mask, n);
}

void POCSnonnegative(MultidimArray<double> &I) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
		DIRECT_MULTIDIM_ELEM(I, n) = std::max(0.0, DIRECT_MULTIDIM_ELEM(I, n));
}

void POCSFourierAmplitude(const MultidimArray<double> &V1FourierMag,
		MultidimArray<std::complex<double>> &V2Fourier, double l) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V1FourierMag) {
		double mod = std::abs(DIRECT_MULTIDIM_ELEM(V2Fourier, n));
		if (mod > 1e-10) // Condition to avoid divide by zero, values smaller than
			// this threshold are considered zero
			DIRECT_MULTIDIM_ELEM(V2Fourier, n) *=
					((1 - l) + l * DIRECT_MULTIDIM_ELEM(V1FourierMag, n)) / mod;
	}
}

void POCSFourierAmplitudeRadAvg(MultidimArray<std::complex<double>> &V,
		double l, const MultidimArray<double> &rQ,
		int V1size_x, int V1size_y, int V1size_z) {
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

void POCSMinMax(MultidimArray<double> &V, double v1m, double v1M) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V) {
		double val = DIRECT_MULTIDIM_ELEM(V, n);
		if (val < v1m)
			DIRECT_MULTIDIM_ELEM(V, n) = v1m;
		else if (val > v1M)
			DIRECT_MULTIDIM_ELEM(V, n) = v1M;
	}
}

void POCSFourierPhase(const MultidimArray<std::complex<double>> &phase,
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

void radialAverage(const MultidimArray<double> &VolFourierMag,
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

MultidimArray<double> computeRadQuotient(const MultidimArray<double> &v1Mag,
		const MultidimArray<double> &vMag, const MultidimArray<double> &V1,
		const MultidimArray<double> &V) {
	// Compute the quotient of the radial mean of the volumes to use it in POCS amplitude
	MultidimArray<double> radial_meanV1;
	radialAverage(v1Mag, V1, radial_meanV1);
	MultidimArray<double> radial_meanV;
	radialAverage(vMag, V, radial_meanV);
	MultidimArray<double> radQuotient = radial_meanV1 / radial_meanV;
	FOR_ALL_ELEMENTS_IN_ARRAY1D(radQuotient)
	radQuotient(i) = std::min(radQuotient(i), 1.0);
	return radQuotient;
}

void createFilter(FourierFilter &filter2, double cutFreq) {
	filter2.FilterBand = LOWPASS;
	filter2.FilterShape = RAISED_COSINE;
	filter2.raised_w = 0.02;
	filter2.w1 = cutFreq;
}

Image<double> subtraction(Image<double> V1, Image<double> &V,
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

MultidimArray<double> computeMagnitude(MultidimArray<double> &volume) {
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
		mask2.read(fnM2);
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

/* Core of the program: processing needed to adjust input
 * volume V2 to reference volume V1. Several iteration of
 * this processing should be run. */
void ProgVolumeSubtraction::runIteration(Image<double> &V,Image<double> &V1,const MultidimArray<double> &radQuotient,
		const MultidimArray<double> &V1FourierMag,const MultidimArray<std::complex<double>> &V2FourierPhase,
		const MultidimArray<double> &mask, FourierFilter &filter2) {
	if (computeE)

	transformer2.FourierTransform(V(), V2Fourier, false);
	if (radavg) {
		auto V1size_x = (int)XSIZE(V1());
		auto V1size_y = (int)YSIZE(V1());
		auto V1size_z = (int)ZSIZE(V1());
		POCSFourierAmplitudeRadAvg(V2Fourier, lambda, radQuotient, V1size_x, V1size_y, V1size_z);
	}
	else {
		POCSFourierAmplitude(V1FourierMag, V2Fourier, lambda);
	}
	transformer2.inverseFourierTransform();
	if (computeE) {
		computeEnergy(Vdiff(), V());
		Vdiff = V;
	}
	POCSMinMax(V(), v1min, v1max);
	if (computeE) {
		computeEnergy(Vdiff(), V());
		Vdiff = V;
	}
	POCSmask(mask, V());
	if (computeE) {
		computeEnergy(Vdiff(), V());
		Vdiff = V;
	}
	transformer2.FourierTransform();
	POCSFourierPhase(V2FourierPhase, V2Fourier);
	transformer2.inverseFourierTransform();
	if (computeE) {
		computeEnergy(Vdiff(), V());
		Vdiff = V;
	}
	POCSnonnegative(V());
	if (computeE) {
		computeEnergy(Vdiff(), V());
		Vdiff = V;
	}
	double std2 = V().computeStddev();
	V() *= std1 / std2;
	if (computeE) {
		computeEnergy(Vdiff(), V());
		Vdiff = V;
	}

	if (cutFreq != 0) {
		filter2.generateMask(V());
		filter2.do_generate_3dmask = true;
		filter2.applyMaskSpace(V());
		if (computeE) {
			computeEnergy(Vdiff(), V());
			Vdiff = V;
		}
	}
}

void ProgVolumeSubtraction::run() {
	show();
	Image<double> V1;
	V1.read(fnVol1);
	auto mask = createMask(V1, fnMask1, fnMask2);
	POCSmask(mask, V1());
	POCSnonnegative(V1());
	V1().computeDoubleMinMax(v1min, v1max);
	Image<double> V;
	V.read(fnVol2);
	POCSmask(mask, V());
	POCSnonnegative(V());
	std1 = V1().computeStddev();
	Vdiff = V;
	auto V2FourierPhase = computePhase(V());
	auto V1FourierMag = computeMagnitude(V1());
	auto V2FourierMag = computeMagnitude(V());
	auto radQuotient = computeRadQuotient(V1FourierMag, V2FourierMag, V1(), V());
	FourierFilter filter2;
	createFilter(filter2, cutFreq);
	for (n = 0; n < iter; ++n) {
		runIteration(V, V1, radQuotient, V1FourierMag, V2FourierPhase, mask, filter2);
	}
	if (performSubtraction) {
		auto masksub = getSubtractionMask(fnMaskSub, mask);
		V1.read(fnVol1);
		V = subtraction(V1, V, masksub, fnVol1F, fnVol2A, filter2, cutFreq);
	}
	/* The output of this program is either a modified
	 * version of V (V') or the subtraction between
	 * V1 and V' if performSubtraction flag is activated' */
	V.write(fnOutVol);
}
