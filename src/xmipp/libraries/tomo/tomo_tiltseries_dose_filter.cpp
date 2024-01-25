/***************************************************************************
 *
 * Authors:    Roberto Marabini       roberto@cnb.csic.es   
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

#include "tomo_tiltseries_dose_filter.h"
#include "core/xmipp_image_generic.h"

#include "data/fourier_filter.h"

#define OUTSIDE_WRAP 0
#define OUTSIDE_AVG 1
#define OUTSIDE_VALUE 2

void ProgTomoTSFilterDose::readParams() {
	fnTs = getParam("-i");
	fnOut = getParam("-o");
	sampling = getDoubleParam("--sampling");
	accVoltage = getDoubleParam("--voltage");
}


void ProgTomoTSFilterDose::show()
{
	if (!verbose)
		return;
	std::cout << "Input movie:           " << fnTs
			<< std::endl << "Output movie:          "
			<< fnOut << std::endl
			<< "Sampling:              " << sampling << std::endl
			<< "Acceleration_Voltage (kV):   " << accVoltage << std::endl;
}


void ProgTomoTSFilterDose::defineParams() {
	addUsageLine("AThis algorithm applies a dose filter based on Grant & Grigorief eLife 2015. The result will be a low pass filtered tilt series based on the applied dose.");
	addParamsLine("   -i <tiltseries>                  : Metadata with the input tilt series");
	addParamsLine("  [-o <fn=\"out.mrcs\">]            : Output path for the filtered tilt series.");
	addParamsLine("  [--sampling <Ts=1>]          	   : Sampling rate (A/pixel)");
	addParamsLine("  [--voltage <voltage=300>] 		   : Acceleration voltage (kV) min_value=200.0e0,max_value=300.0e0)");
	addExampleLine("Example", false);
	addExampleLine("xmipp_tomo_tiltseries_dose_filter -i tiltseries.xmd -o . --sampling 2.2 --dosePerImage 2 --voltage 300");
	addExampleLine("xmipp_tomo_tiltseries_dose_filter -i tiltseries.xmd -o . --sampling 2.2 --voltage 300");
}


double ProgTomoTSFilterDose::doseFilter(double dose_at_end_of_frame, double critical_dose)
{
	return exp((-0.5 * dose_at_end_of_frame) / critical_dose);
}


double ProgTomoTSFilterDose::criticalDose(double spatial_frequency)
{
	return ((critical_dose_a * pow(spatial_frequency, critical_dose_b)) +  critical_dose_c) * voltage_scaling_factor;
}

void ProgTomoTSFilterDose::initParameters()
{
	/**  Set up the critical curve function (summovie) */
	critical_dose_a = 0.24499;
	critical_dose_b = -1.6649;
	critical_dose_c = 2.8141;
	// init with a very large number
	critical_dose_at_dc = std::numeric_limits<double>::max() * 0.001;

	// Defining the scaling factor given the acceleration voltage. The methods considers only two voltage 200keV and 300keV
	// Any other voltage would result in an error
	if (accVoltage < 301 && accVoltage > 299.)
	{
			accVoltage = 300.0;
			voltage_scaling_factor = 1.0;
	}
	else
	{
		if (accVoltage < 201.0 && accVoltage > 199.0)
		{
			accVoltage = 200.0;
			voltage_scaling_factor = 0.8;
		}
		else
		{
			REPORT_ERROR(ERR_ARG_INCORRECT, "Bad acceleration voltage (must be 200 or 300 kV");
		}
			
	}

}

void ProgTomoTSFilterDose::applyDoseFilterToImage(int Ydim, int Xdim, const MultidimArray<std::complex<double> > &FFT1, const double dose_finish)
{
	double ux, uy;
	double yy;
	double current_critical_dose, current_optimal_dose;
	int sizeX_2 = Xdim / 2;
	double ixsize = 1.0 / Xdim;
	int sizeY_2 = Ydim / 2;
	double iysize = 1.0 / Ydim;

	for (long int i = 0; i < YSIZE(FFT1); i++) //i->y,j->x xmipp convention is oposite summove
	{ 
		FFT_IDX2DIGFREQ_FAST(i, Ydim, sizeY_2, iysize, uy);
		
		yy = uy * uy;
		for (long int j = 0; j < XSIZE(FFT1); j++)
		{
			FFT_IDX2DIGFREQ_FAST(j, Xdim, sizeX_2, ixsize, ux);

			if (i == 0 && j == 0)
			{
				current_critical_dose = critical_dose_at_dc;
			}
			else
			{
				current_critical_dose = criticalDose(sqrt(ux * ux + yy) / sampling);
			}

		    current_optimal_dose = 2.51284 * current_critical_dose; //This is the optimal dose given the critical dose (see Grant, Grigorief eLife 2015)

			DIRECT_A2D_ELEM(FFT1, i, j ) *= doseFilter(dose_finish, current_critical_dose);
				

		}
	}
}


void ProgTomoTSFilterDose::readInputData(MetaDataVec &mdts)
{
	size_t Xdim, Ydim, Zdim, Ndim;

	//if input is an stack create a metadata.
	if (fnTs.isMetaData())
		mdts.read(fnTs);
	else
	{
		REPORT_ERROR(ERR_ARG_INCORRECT, "The input must be a metadata with the image filenames and the accumulated dose");
	}
}


void ProgTomoTSFilterDose::run()
{
	
	show();

	initParameters();
	
	MetaDataVec mdts;

	readInputData(mdts);

	std::cout << " Starting filtering..."<< std::endl;

	
	FourierTransformer transformer;
	FileName fnti;
	FileName fnOutFrame;
	
	Image<double> tiltImage;
	MultidimArray<double> &ptrtiltImage = tiltImage();

	int mode = WRITE_OVERWRITE;
	double doseTi;

	FileName fnk;
	std::vector<FileName> ti_fn;
	std::vector<double> ti_dose;

	for (size_t objId : mdts.ids())
	{
		mdts.getValue(MDL_IMAGE, fnti, objId);

		mdts.getValue(MDL_DOSE, doseTi, objId);

		ti_fn.push_back(fnti);
		ti_dose.push_back(doseTi);
	}

	FileName fnTsOut;

	size_t n = 0;
	MultidimArray<std::complex<double> > FFT1;
	
	for (size_t objId : mdts.ids())
	{
		mdts.getValue(MDL_IMAGE, fnti, objId);
		mdts.getValue(MDL_DOSE, doseTi, objId);
		
		tiltImage.read(fnti);

		// Now do the Fourier transform and filter
		transformer.FourierTransform(ptrtiltImage, FFT1, false);

		applyDoseFilterToImage((int) YSIZE(ptrtiltImage), (int) XSIZE(ptrtiltImage), FFT1, doseTi);

		transformer.inverseFourierTransform();

		fnTsOut = fnOut;
		tiltImage.write(fnTsOut, n+FIRST_IMAGE, true, mode);
		mode = WRITE_APPEND;

		++n;

	}
}