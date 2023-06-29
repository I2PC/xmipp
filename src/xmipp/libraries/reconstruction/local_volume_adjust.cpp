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

#include "local_volume_adjust.h"
#include "core/transformations.h"
#include <core/histogram.h>
#include <core/xmipp_fftw.h>
#include <core/xmipp_program.h>
#include <data/fourier_filter.h>


// Usage ===================================================================
void ProgLocalVolumeAdjust::defineParams() {
	// Usage
	addUsageLine("This program modifies a volume in order to adjust its intensity locally to a reference volume");
	// Parameters
	addParamsLine("--i1 <volume>			: Reference volume");
	addParamsLine("--i2 <volume>			: Volume to modify");
	addParamsLine("[-o <structure=\"\">]\t: Volume 2 modified or "
			"volume difference");
	addParamsLine("\t: If no name is given, "
			"then output_volume.mrc");
	addParamsLine("[--sub]\t: Perform the "
			"subtraction of the volumes. Output will be the difference");
	addParamsLine("[--mask1 <mask=\"\">]		: Mask for volume 1");
}

// Read arguments ==========================================================
void ProgLocalVolumeAdjust::readParams() {
	fnVol1 = getParam("--i1");
	fnVol2 = getParam("--i2");
	fnOutVol = getParam("-o");
	if (fnOutVol.isEmpty())
		fnOutVol = "output_volume.mrc";
	performSubtraction = checkParam("--sub");
	fnMask1 = getParam("--mask1");
}

// Show ====================================================================
void ProgLocalVolumeAdjust::show() const {
	std::cout << "Input volume 1:\t" << fnVol1 << std::endl
			<< "Input volume 2:\t" << fnVol2 << std::endl
			<< "Input mask 1:\t" << fnMask1 << std::endl
			<< "Output:\t" << fnOutVol << std::endl;
}

void ProgLocalVolumeAdjust::run() {
	show();
    Image<double> V1;
	V1.read(fnVol1);
	Image<double> V;
	V.read(fnVol2);
	/* The output of this program is a modified version of V (V')*/
	V.write(fnOutVol);
}