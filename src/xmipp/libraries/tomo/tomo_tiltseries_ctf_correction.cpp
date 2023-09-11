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

#include "tomo_tiltseries_ctf_correction.h"
#include "core/xmipp_image_generic.h"
#include <core/metadata_vec.h>
#include "data/fourier_filter.h"
#include "readData_utils.h"


void ProgTomoTSCTFCorrection::readParams()
{
	fnTs = getParam("-i");
	fnOut = getParam("-o");
	sampling = getDoubleParam("--sampling");
}


void ProgTomoTSCTFCorrection::show()
{
	if (!verbose)
		return;
	std::cout << "Input tilt series:           " << fnTs
			  << std::endl << "Output tilt series:          "
			  << fnOut << std::endl
			  << "Sampling:              " << sampling << std::endl;
}


void ProgTomoTSCTFCorrection::defineParams() {
	addUsageLine("This algorithm applies CTF correction in the tilt series by means of a Wiener filter.");
	addParamsLine("   -i <tiltseries>                  : Metadata with the input tilt series");
	addParamsLine("  [-o <fn=\"out.mrcs\">]            : Output path for the filtered tilt series.");
	addParamsLine("  [--sampling <Ts=1>]          	   : Sampling rate (A/pixel)");
	addExampleLine("Example", false);
	addExampleLine("xmipp_tomo_tiltseries_ctf_correction -i tiltseries.xmd -o . --sampling 2.2 ");
}


void ProgTomoTSCTFCorrection::run()
{
	MetaDataVec mdts;
	String message;
	message = "The input -i must be a metadata file .xmd with the list of tilt series.";
	readInputData(mdts, fnTs, message);

	Image<double> img;
	auto &ptrImg = img();

	WF.pad = 1;
	WF.correct_envelope = correct_envelope;
	WF.sampling_rate = sampling_rate;
	WF.wiener_constant = wiener_constant;
	WF.isIsotropic = isIsotropic;
	WF.phase_flipped = phase_flipped;

	double tilt;

	for (const auto& row : mdts)
	{
		FileName fnTi;
		row.getValue(MDL_IMAGE, fnTi);
		row.getValue(MDL_IMAGE, tilt);


		img.read(fnTi);

		CTFDescription ctf;
		ctf.readFromMdRow(row);
		ctf.phase_shift = (ctf.phase_shift*PI)/180;

		WF.applyWienerFilter(ptrImg, ctf);

	}


}


