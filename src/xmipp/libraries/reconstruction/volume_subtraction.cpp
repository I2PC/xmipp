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
	addParamsLine("[-o <structure=\"\">]		: Volume 2 modified or "
			"volume difference");
	addParamsLine("					: If no name is given, "
			"then output_volume.mrc");
	addParamsLine("[--sub]				: Perform the "
			"subtraction of the volumes. Output will be the difference");
	addParamsLine("[--sigma <s=3>]			: Decay of the filter "
			"(sigma) to smooth the mask transition");
	addParamsLine(
			"[--iter <n=5>]\t: Number of iterations for the adjustment process");
	addParamsLine("[--mask1 <mask=\"\">]		: Mask for volume 1");
	addParamsLine("[--mask2 <mask=\"\">]		: Mask for volume 2");
	addParamsLine(
			"[--maskSub <mask=\"\">]		: Mask for subtraction region");
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

// Variables to store read parameters
FileName fnVol1, fnVol1F, fnVol2A, fnMaskSub, fnMask1, fnMask2, fnOutVol;
bool performSubtraction;
int sigma;
double cutFreq;
double lambda;
bool saveVol1Filt;
bool saveVol2Adj;
bool radavg;

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
	saveVol1Filt = checkParam("--saveV1");
	saveVol2Adj = checkParam("--saveV2");
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
	std::cout << "Input volume 1:    		" << fnVol1 << std::endl
			<< "Input volume 2:    	   	" << fnVol2 << std::endl
			<< "Input mask 1:    	   	" << fnMask1 << std::endl
			<< "Input mask 2:    	   	" << fnMask2 << std::endl
			<< "Input mask sub:		" << fnMaskSub << std::endl
			<< "Sigma:			" << sigma << std::endl
			<< "Iterations:			" << iter << std::endl
			<< "Cutoff frequency:		" << cutFreq << std::endl
			<< "Relaxation factor:		" << lambda << std::endl
			<< "Match radial averages:\t" << radavg << std::endl
			<< "Output:			" << fnOutVol << std::endl;
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

void POCSFourierAmplitude(const MultidimArray<double> &A,
		MultidimArray<std::complex<double>> &FI,
		double lambda) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(A) {
		double mod = std::abs(DIRECT_MULTIDIM_ELEM(FI, n));
		if (mod > 1e-10) // Condition to avoid divide by zero, values smaller than
			// this threshold are considered zero
			DIRECT_MULTIDIM_ELEM(FI, n) *=
					((1 - lambda) + lambda * DIRECT_MULTIDIM_ELEM(A, n)) / mod;
	}
}

void POCSFourierAmplitudeRadAvg(MultidimArray<std::complex<double>> &V,
		double lambda, const MultidimArray<double> &rQ,
		int V1size_x, int V1size_y, int V1size_z) {
	int V1size2_x = V1size_x / 2;
	double V1sizei_x = 1.0 / V1size_x;
	int V1size2_y = V1size_y / 2;
	double V1sizei_y = 1.0 / V1size_y;
	int V1size2_z = V1size_z / 2;
	double V1sizei_z = 1.0 / V1size_z;
	double wx;
	double wy;
	double wz;
	for (int k = 0; k < V1size_z; ++k) {
		FFT_IDX2DIGFREQ_FAST(k, V1size_z, V1size2_z, V1sizei_z, wz)
    		  double wz2 = wz * wz;
		for (int i = 0; i < V1size_y; ++i) {
			FFT_IDX2DIGFREQ_FAST(i, V1size_y, V1size2_y, V1sizei_y, wy)
        		double wy2 = wy * wy;
			for (int j = 0; j < V1size_x; ++j) {
				FFT_IDX2DIGFREQ_FAST(j, V1size_x, V1size2_x, V1sizei_x, wx)
        		  double w = sqrt(wx * wx + wy2 + wz2);
				auto iw = (int)round(w * V1size_x);
				DIRECT_A3D_ELEM(V, k, i, j) *=
						(1 - lambda) + lambda * DIRECT_MULTIDIM_ELEM(rQ, iw);
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
void extractPhase(MultidimArray<std::complex<double>> &FI) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FI) {
		double *ptr = (double *)&DIRECT_MULTIDIM_ELEM(FI, n);
		double phi = atan2(*(ptr + 1), *ptr);
		DIRECT_MULTIDIM_ELEM(FI, n) = std::complex<double>(cos(phi), sin(phi));
	}
}

void computeEnergy(MultidimArray<double> &Vdiff, MultidimArray<double> &Vact) {
	Vdiff = Vdiff - Vact;
	double energy = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Vdiff)
	energy += DIRECT_MULTIDIM_ELEM(Vdiff, n) * DIRECT_MULTIDIM_ELEM(Vdiff, n);
	energy = sqrt(energy / MULTIDIM_SIZE(Vdiff));
	std::cout << "Energy: " << energy << std::endl;
}

void centerFFTMagnitude(MultidimArray<double> &VolRad,
		MultidimArray<std::complex<double>> &VolFourierRad,
		MultidimArray<double> &VolFourierMagRad) {
	FourierTransformer transformerRad;
	transformerRad.completeFourierTransform(VolRad, VolFourierRad);
	CenterFFT(VolFourierRad, true);
	FFT_magnitude(VolFourierRad, VolFourierMagRad);
	VolFourierMagRad.setXmippOrigin();
}

void radialAverage(const MultidimArray<double> &VolFourierMagRad,
		const MultidimArray<std::complex<double>> &VolFourierRad,
		MultidimArray<double> const &Volrad,
		MultidimArray<double> radial_mean) {
	Matrix1D<int> center(2);
	center.initZeros();
	MultidimArray<double> radial_mean;
	MultidimArray<int> radial_count;
	radialAverageNonCubic(VolFourierMagRad, center, radial_mean, radial_count);
	FOR_ALL_ELEMENTS_IN_ARRAY1D(VolFourierRad)
	Volrad(i) = radial_mean(i);
}

MultidimArray<double> computeRadialMean(MultidimArray<double> volume) {
	volume.setXmippOrigin();
	MultidimArray<std::complex<double>> fourierRad;
	MultidimArray<double> fourierMagRad;
	centerFFTMagnitude(volume, fourierRad, fourierMagRad);
	MultidimArray<double> radialMean;
	radialAverage(fourierMagRad, fourierRad, volume, radialMean);
	return radialMean;
}

MultidimArray<double> computeRadQuotient(const MultidimArray<double> &v1,
		const MultidimArray<double> &v) {
	// Compute the quotient of the radial mean of the volumes to use it in POCS amplitude
	MultidimArray<double> radQuotient;
	MultidimArray<double> radial_meanV1 = computeRadialMean(v1);
	MultidimArray<double> radial_meanV = computeRadialMean(v);
	radQuotient = radial_meanV1 / radial_meanV;
	FOR_ALL_ELEMENTS_IN_ARRAY1D(radQuotient)
	radQuotient(i) = std::min(radQuotient(i), 1.0);

	return radQuotient;
}

void subtraction(MultidimArray<double> &V1,
		const MultidimArray<double> &V1Filtered,
		const MultidimArray<double> &V,
		const MultidimArray<double> &mask) {
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V1)
    		DIRECT_MULTIDIM_ELEM(V1, n) =
    				DIRECT_MULTIDIM_ELEM(V1, n) * (1 - DIRECT_MULTIDIM_ELEM(mask, n)) +
					(DIRECT_MULTIDIM_ELEM(V1Filtered, n) -
							std::min(DIRECT_MULTIDIM_ELEM(V, n),
									DIRECT_MULTIDIM_ELEM(V1Filtered, n))) *
									DIRECT_MULTIDIM_ELEM(mask, n);
}

FourierFilter Filter2;
void createFilter() {
	Filter2.FilterBand = LOWPASS;
	Filter2.FilterShape = RAISED_COSINE;
	Filter2.raised_w = 0.02;
	Filter2.w1 = cutFreq;
}

MultidimArray<double> computeMagnitude(MultidimArray<double> &volume) {
	FourierTransformer transformer;
	MultidimArray<std::complex<double>> fourier;
	MultidimArray<double> magnitude;
	transformer.FourierTransform(volume, fourier, false);
	FFT_magnitude(fourier, magnitude);
	return magnitude;
}

MultidimArray<double> mask;
void createMask(const Image<double> &volume) {
	if (fnMask1 != "" && fnMask2 != "") {
		Image<double> mask1;
		Image<double> mask2;
		mask1.read(fnMask1);
		mask2.read(fnMask2);
		mask = mask1() * mask2();
	} else {
		mask.resizeNoCopy(volume());
		mask.initConstant(1.0);
	}
}

void filterMask(MultidimArray<double> &mask) {
	FourierFilter Filter;
	Filter.FilterShape = REALGAUSSIAN;
	Filter.FilterBand = LOWPASS;
	Filter.w1 = sigma;
	Filter.applyMaskSpace(mask);
}

FourierTransformer transformer2;
MultidimArray<std::complex<double>> computePhase(MultidimArray<double> &volume) {
	MultidimArray<std::complex<double>> phase;
	transformer2.FourierTransform(volume, phase, true);
	extractPhase(phase);
	return phase;
}

/* The output of this program is either a modified
 * version of V2 (V2') or the subtraction between
 * V1 and V2 if performSubtraction flag is activated' */
void writeResults(Image<double> &V, Image<double> &V1,
		MultidimArray<double> &mask) {
	if (performSubtraction) {
		Image<double> V1Filtered;
		V1.read(fnVol1);
		V1Filtered() = V1();
		if (cutFreq != 0){
			Filter2.applyMaskSpace(V1Filtered());
		}
		if (saveVol1Filt) {
			V1Filtered.write(fnVol1F);
		}
		if (saveVol2Adj) {
			V.write(fnVol2A);
		}
		if (fnMaskSub.isEmpty()) {
			filterMask(mask);
		}
		subtraction(V1(), V1Filtered(), V(), mask);
		V1.write(fnOutVol);
	} else {
		V.write(fnOutVol);
	}
}

// Declaration of variables needed to perform an iteration
double v1min;
double v1max;
MultidimArray<std::complex<double>> V2Fourier;

/* Core of the program: processing needed to adjust input
 * volume V2 to reference volume V1. Several iteration of
 * this processing should be run. */
template <bool computeE>
void runIteration(size_t n, Image<double> &V, Image<double> &Vdiff,
		Image<double> &V1, MultidimArray<double> &radQuotient,
		MultidimArray<double> &V1FourierMag, double std1,
		const MultidimArray<std::complex<double>> &V2FourierPhase) {
	if (computeE)
		std::cout << "---Iter " << n << std::endl;
	if (radavg) {
		auto V1size_x = (int)XSIZE(V1());
		auto V1size_y = (int)YSIZE(V1());
		auto V1size_z = (int)ZSIZE(V1());
		transformer2.completeFourierTransform(V(), V2Fourier);
		CenterFFT(V2Fourier, true);
		POCSFourierAmplitudeRadAvg(V2Fourier, lambda, radQuotient, V1size_x,
				V1size_y, V1size_z);
	} else {
		transformer2.FourierTransform(V(), V2Fourier, false);
	}
	POCSFourierAmplitude(V1FourierMag, V2Fourier, lambda);
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
		Filter2.generateMask(V());
		Filter2.do_generate_3dmask = true;
		Filter2.applyMaskSpace(V());
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
	createMask(V1);
	POCSmask(mask, V1());
	POCSnonnegative(V1());
	V1().computeDoubleMinMax(v1min, v1max);

	Image<double> V;
	V.read(fnVol2);
	POCSmask(mask, V());
	POCSnonnegative(V());

	double std1 = V1().computeStddev();
	Image<double> Vdiff = V;
	auto V2FourierPhase = computePhase(V());
	auto radQuotient = computeRadQuotient(V1(), V());
	auto V1FourierMag = computeMagnitude(V1());
	createFilter();

	for (size_t n = 0; n < iter; ++n) {
		computeE ? runIteration<true>(n, V, Vdiff, V1, radQuotient, V1FourierMag,
				std1, V2FourierPhase)
				: runIteration<false>(n, V, Vdiff, V1, radQuotient, V1FourierMag,
						std1, V2FourierPhase);
	}
	if (!fnMaskSub.isEmpty()) {
		Image<double> tmp(mask);
		tmp.read(fnMaskSub);
	}
	writeResults(V1, V, mask);
}
