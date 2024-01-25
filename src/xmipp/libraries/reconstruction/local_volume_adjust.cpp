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
	addParamsLine("[-o <structure=\"\">]\t: Volume 2 modified or volume difference");
	addParamsLine("\t: If no name is given, then output_volume.mrc");
	addParamsLine("--mask <mask=\"\">		: Mask for volume 1");
	addParamsLine("[--sampling <sampling=1>]\t: Sampling rate (A/pixel)");
	addParamsLine("[--neighborhood <n=5>]\t: side length (in Angstroms) of a square which will define the region of adjustment");
	addParamsLine("[--sub]\t: Perform the subtraction of the volumes. Output will be the difference");
	addParamsLine("--save <structure=\"\">\t: Path for saving occupancy volume"); 
}

// Read arguments ==========================================================
void ProgLocalVolumeAdjust::readParams() {
	fnVol1 = getParam("--i1");
	fnVol2 = getParam("--i2");
	fnOutVol = getParam("-o");
	if (fnOutVol.isEmpty())
		fnOutVol = "output_volume.mrc";
	performSubtraction = checkParam("--sub");
	fnMask = getParam("--mask");
	sampling = getDoubleParam("--sampling");
	neighborhood = getIntParam("--neighborhood"); 
	fnOccup = getParam("--save"); 
}

// Show ====================================================================
void ProgLocalVolumeAdjust::show() const {
	std::cout << "Input volume 1 (reference):\t" << fnVol1 << std::endl
			<< "Input volume 2:\t" << fnVol2 << std::endl
			<< "Input mask:\t" << fnMask << std::endl
			<< "Output:\t" << fnOutVol << std::endl;
}

void ProgLocalVolumeAdjust::run() {
	show();
	// Read inputs
    Image<double> Vref;
	Vref.read(fnVol1);
	MultidimArray<double> &mVref=Vref();
	mVref.setXmippOrigin();
	Image<double> V;
	V.read(fnVol2);
	MultidimArray<double> &mV=V();
	mV.setXmippOrigin();
	Image<double> M;
	M.read(fnMask);
	MultidimArray<double> &mM=M();
	mM.setXmippOrigin();
	int iters;
	int neighborhood_px;
	neighborhood_px = round(neighborhood/sampling);
	int cubic_neighborhood;
	cubic_neighborhood = neighborhood_px*neighborhood_px*neighborhood_px;
	iters = floor(ZYXSIZE(mV)/cubic_neighborhood);
	int xsize = XSIZE(mV);
	int ysize = YSIZE(mV);
	int zsize = ZSIZE(mV);
	int ki = 0;
	int ii = 0;
	int ji = 0;
	// Initialize occupancy volume
	Image<double> Voccupancy;
	Voccupancy = Vref;
	MultidimArray<double> &mVoc=Voccupancy();
	mVoc.initZeros();

	for (size_t s=0; s < iters; s++) 
	{
		sumV_Vref = 0;
		sumVref2 = 0;
		// Go over each subvolume
		for (k=0; k < neighborhood_px; ++k)
		{
			for (i=0; i < neighborhood_px; ++i)
			{
				for (j=0; j < neighborhood_px; ++j)
				{
					if (DIRECT_A3D_ELEM(mM,ki+k,ii+i,ji+j) == 1) // Condition to check if we are inside mask
					{
						sumV_Vref += DIRECT_A3D_ELEM(mV,ki+k,ii+i,ji+j)*DIRECT_A3D_ELEM(mVref,ki+k,ii+i,ji+j);
						sumVref2 += DIRECT_A3D_ELEM(mVref,ki+k,ii+i,ji+j)*DIRECT_A3D_ELEM(mVref,ki+k,ii+i,ji+j);
					}
					else 
					{
						sumV_Vref += 0;
						sumVref2 += 0;
					}
				}     
			}
		}
		// Compute constant (c)
		if (sumVref2 == 0)
			c = 0;
		else
			c = sumV_Vref/sumVref2;
		
		// Apply adjustment per regions 
		for (k=0; k < neighborhood_px; ++k)
		{
			for (i=0; i < neighborhood_px; ++i)
			{
				for (j=0; j < neighborhood_px; ++j)
				{
					if (DIRECT_A3D_ELEM(mM,ki+k,ii+i,ji+j) == 1) // Condition to check if we are inside mask
					{
						DIRECT_A3D_ELEM(mV,ki+k,ii+i,ji+j) /= c;
						// construct occupancy volume
						DIRECT_A3D_ELEM(mVoc,ki+k,ii+i,ji+j) = c;
					}
				}
			}
		}
		// Take the index to start in next subvolume
		if (ji < (xsize-neighborhood_px))
			ji += neighborhood_px;
		else
			ji = 0;
		if (ji == 0)
		{
			if (ii < (ysize-neighborhood_px))
				ii += neighborhood_px;
			else
				ii = 0;
		}
		if (ii == 0 && ji == 0)
		{
			if (ki < (zsize-neighborhood_px))
				ki += neighborhood_px;
			else
				ki = 0;
		}
	}
	// Save occupancy volume
	Voccupancy.write(formatString("%s/Occupancy.mrc", fnOccup.c_str()));

	// The output of this program is a modified version of V (V')
	if (performSubtraction) // Or the output is the subtraction V = Vref - V
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mV)
			DIRECT_MULTIDIM_ELEM(mV, n) = DIRECT_MULTIDIM_ELEM(mVref, n) * (1 - DIRECT_MULTIDIM_ELEM(mM, n)) 
				+ (DIRECT_MULTIDIM_ELEM(mVref, n) - std::min(DIRECT_MULTIDIM_ELEM(mV, n), DIRECT_MULTIDIM_ELEM(mVref, n))) 
				* DIRECT_MULTIDIM_ELEM(mM, n);
	}
	V.write(fnOutVol);
}