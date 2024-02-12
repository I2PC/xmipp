/***************************************************************************
 *
 * Authors:    Federico P. de Isidro Gomez			  fp.deisidro@cnb.csic.es
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

#include "tomo_tiltseries_detect_misalignment_corr.h"
#include <chrono>



// --------------------------- INFO functions ----------------------------

void ProgTomoTSDetectMisalignmentCorr::readParams()
{
	fnVol = getParam("-i");
	fnTiltAngles = getParam("--tlt");
	fnOut = getParam("-o");

	samplingRate = getDoubleParam("--samplingRate");
	targetFS = getDoubleParam("--targetLMsize");
}



void ProgTomoTSDetectMisalignmentCorr::defineParams()
{
	addUsageLine("This function determines the location of high contrast features in a volume.");
	addParamsLine("  -i <mrcs_file=\"\">                   					: Input tilt-series.");
	addParamsLine("  --tlt <xmd_file=\"\">      							: Input file containning the tilt angles of the tilt-series in .xmd format.");
	addParamsLine("  --inputCoord <output=\"\">								: Input coordinates of the 3D landmarks. Origin at top left coordinate (X and Y always positive) and centered at the middle of the volume (Z positive and negative).");
	addParamsLine("  [-o <output=\"./alignemntReport.xmd\">]       			: Output file containing the alignemnt report.");

	addParamsLine("  [--samplingRate <samplingRate=1>]						: Sampling rate of the input tomogram (A/px).");

	addParamsLine("  [--targetLMsize <targetLMsize=8>]		    : Targer size of landmark when downsampling (px).");
}



void ProgTomoTSDetectMisalignmentCorr::generateSideInfo()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Generating side info..." << std::endl;
	#endif

	// Read tilt angles file
	MetaDataVec inputTiltAnglesMd;
	inputTiltAnglesMd.read(fnTiltAngles);

	size_t objIdTlt;

	#ifdef VERBOSE_OUTPUT
	std::cout << "Input tilt angles read from: " << fnTiltAngles << std::endl;
	#endif

	// Initialize local alignment vector (depends on the number of acquisition angles)
	localAlignment.resize(nSize, true);

	#ifdef VERBOSE_OUTPUT
	std::cout << "Side info generated succesfully!" << std::endl;
	#endif
}


// --------------------------- HEAD functions ----------------------------

void ProgTomoTSDetectMisalignmentCorr::detectSubtleMisalingment(MultidimArray<double> &ts)
{
	MultidimArray<double> ti_bw;
	MultidimArray<double> ti_fw;

	ti_bw.initZeros(ySize, xSize);
	ti_fw.initZeros(ySize, xSize);

	double ta_bw;
	double ta_fw;

	std::vector<Matrix2D<double>> shifts_bw(nSize-1);
	std::vector<Matrix2D<double>> shifts_fw(nSize-1);

	// Forward 
	for (size_t n = 0; n < nSize-1; n++)
	{
		// Construct forward and backward images
		for (size_t i = 0; i < ySize; i++)
		{
			for (size_t j = 0; j < xSize; j++)
			{
				DIRECT_A2D_ELEM(ti_bw, i, j) = DIRECT_NZYX_ELEM(ts, n, 0, i, j);
				DIRECT_A2D_ELEM(ti_fw, i, j) = DIRECT_NZYX_ELEM(ts, n+1, 0, i, j);
			}
		}

		// Apply cosine stretching
		ta_bw = tiltAngles[n];
		ta_fw = tiltAngles[n+1];

		int direction;

		if (abs(ta_bw)>abs(ta_fw))
		{
			cosineStretching(ti_bw, ta_bw-ta_fw);
			direction = 1;
		}
		else
		{
			cosineStretching(ti_fw, ta_bw-ta_fw);
			direction = -1;
		}

		// Calculate shift for maximum correlation
		Matrix2D<double> relShift;
		relShift = maxCorrelationShift(ti_bw, ti_fw);

		MAT_ELEM(relShift, 0, 0) *= direction;
		MAT_ELEM(relShift, 0, 1) *= direction;


		std::cout << "For image " << n << ": shift [" << MAT_ELEM(relShift, 0, 0) << ", " << MAT_ELEM(relShift, 0, 1) << "]" << std::endl;

		relativeShifts.push_back(relShift);
	}
}



// --------------------------- I/O functions ----------------------------



// --------------------------- MAIN ----------------------------------
void ProgTomoTSDetectMisalignmentCorr::run()
{
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;	

	auto t1 = high_resolution_clock::now();

	std::cout << "Starting..." << std::endl;

	size_t Xdim, Ydim;

	MetaDataVec tiltseriesmd;
    Image<double> tiltSeriesImages;

    if (fnVol.isMetaData())
    {
        tiltseriesmd.read(fnVol);
    }
    else
    {
        tiltSeriesImages.read(fnVol, HEADER);

        size_t Zdim, Ndim;
        tiltSeriesImages.getDimensions(Xdim, Ydim, Zdim, Ndim);

        if (fnVol.getExtension() == "mrc" and Ndim == 1)
            Ndim = Zdim;

        size_t id;
        FileName fn;
        for (size_t i = 0; i < Ndim; i++) 
        {
            id = tiltseriesmd.addObject();
            fn.compose(i + FIRST_IMAGE, fnVol);
            tiltseriesmd.setValue(MDL_IMAGE, fn, id);
        }
    }

	tiltSeriesImages.getDimensions(xSize, ySize, zSize, nSize);

	#ifdef DEBUG_DIM
	std::cout << "Input tilt-series dimensions:" << std::endl;
	std::cout << "x " << xSize << std::endl;
	std::cout << "y " << ySize << std::endl;
	std::cout << "z " << zSize << std::endl;
	std::cout << "n " << nSize << std::endl;
	#endif

	generateSideInfo();

	detectSubtleMisalingment(tiltSeriesImages());

	auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1); 	// Getting number of milliseconds as an integer
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}


// --------------------------- UTILS functions ----------------------------


void ProgTomoTSDetectMisalignmentCorr::cosineStretching(MultidimArray<double> &ti, double tiltAngle)
{
	Matrix2D<double> sm = getCosineStretchingMatrix(tiltAngle);
	MultidimArray<double> ti_tmp;
	ti_tmp = ti;

    applyGeometry(1, ti, ti_tmp, sm, xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP);

}


Matrix2D<double> ProgTomoTSDetectMisalignmentCorr::getCosineStretchingMatrix(double tiltAngle)
{
	double cosTiltAngle = cos(tiltAngle * PI/180.0);

	Matrix2D<double> m(3,3);

	MAT_ELEM(m, 0, 0) = 1/cosTiltAngle;
	// MAT_ELEM(m, 0, 1) = 0;
	// MAT_ELEM(m, 0, 2) = 0;
	// MAT_ELEM(m, 1, 0) = 0;
	MAT_ELEM(m, 1, 1) = 1;
	// MAT_ELEM(m, 1, 2) = 0;
	// MAT_ELEM(m, 2, 0) = 0;
	// MAT_ELEM(m, 2, 1) = 0;
	MAT_ELEM(m, 2, 2) = 1;

	return m;
}


Matrix2D<double> ProgTomoTSDetectMisalignmentCorr::maxCorrelationShift(MultidimArray<double> &ti1, MultidimArray<double> &ti2)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Calculating shift for maximum correlation..." << std::endl;
	#endif

	Matrix2D<double> relShift(2, 1);
	MultidimArray<double> tsCorr;

	// Shift the particle respect to its symmetric to look for the maximum correlation displacement
	CorrelationAux aux;
	correlation_matrix(ti1, ti2, tsCorr, aux, true);

	auto maximumCorrelation = MINDOUBLE;
	int xDisplacement = 0;
	int yDisplacement = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(tsCorr)
	{
		double value = DIRECT_A2D_ELEM(tsCorr, i, j);

		if (value > maximumCorrelation)
		{
			#ifdef DEBUG_CENTER_COORDINATES
			std::cout << "new maximumCorrelation " << value << " at (" << i << ", " << j << ")" << std::endl;
			#endif

			maximumCorrelation = value;
			xDisplacement = j;
			yDisplacement = i;
		}
	}

	MAT_ELEM(relShift, 0, 0) = (xDisplacement - xSize) / 2;
	MAT_ELEM(relShift, 0, 1) = (yDisplacement - ySize) / 2;


	#ifdef DEBUG_TI_CORR
	std::cout << "maximumCorrelation " << maximumCorrelation << std::endl;
	std::cout << "xDisplacement " << (xDisplacement - xSize) / 2 << std::endl;
	std::cout << "yDisplacement " << (yDisplacement - ySize) / 2 << std::endl;

	std::cout << "Correlation volume dimensions (" << XSIZE(tsCorr) << ", " << YSIZE(tsCorr) << ")" << std::endl;
	#endif

	return relShift;
}