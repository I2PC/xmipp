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
	addParamsLine("[--neighborhood <n=5>]\t: side length (in pixels) of a square which will define the region of adjustment");
	addParamsLine("[--sub]\t: Perform the subtraction of the volumes. Output will be the difference");
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
	neighborhood = getIntParam("--neighborhood"); // TODO: ask in A and convert to pixels (ask sampling rate)?
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
	int cubic_neighborhood;
	cubic_neighborhood = neighborhood*neighborhood*neighborhood;
	iters = floor(ZYXSIZE(mV)/cubic_neighborhood);
	int xsize = XSIZE(mV);
	int ysize = YSIZE(mV);
	int zsize = ZSIZE(mV);
	int ki = 0;
	int ii = 0;
	int ji = 0;
	for (size_t s=0; s < iters; s++) 
	{
		sumV_Vref = 0;
		sumVref2 = 0;
		// Go over each subvolume
		for (k=0; k < neighborhood; ++k)
		{
			for (i=0; i < neighborhood; ++i)
			{
				for (j=0; j < neighborhood; ++j)
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
		std::cout << "c = "<< c << std::endl;
		// Apply adjustment TODO: per regions
		for (k=0; k < neighborhood; ++k)
		{
			for (i=0; i < neighborhood; ++i)
			{
				for (j=0; j < neighborhood; ++j)
				{
					if (DIRECT_A3D_ELEM(mM,ki+k,ii+i,ji+j) == 1) // Condition to check if we are inside mask
					{
						DIRECT_A3D_ELEM(mV,ki+k,ii+i,ji+j) /= c;
					}
				}
			}
		}
		// Take the index to start in next subvolume
		if (ji < (xsize-neighborhood))
			ji += neighborhood;
		else
			ji = 0;
		if (ji == 0)
		{
			if (ii < (ysize-neighborhood))
				ii += neighborhood;
			else
				ii = 0;
		}
		if (ii == 0 && ji == 0)
		{
			if (ki < (zsize-neighborhood))
				ki += neighborhood;
			else
				ki = 0;
		}
	}
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