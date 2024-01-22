/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
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

#ifndef _PROG_TOMO_RESOLUTION_SUBTOMO
#define _PROG_TOMO_RESOLUTION_SUBTOMO

#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <core/xmipp_fftw.h>
#include <limits>
#include <complex>
#include <string>

class ProgTomoResolutionSubtomos : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut;
    FileName fnTomo;
    FileName fnCoor;
    FileName fnHalf;

    size_t Xdim, Xtom;
    size_t Ydim, Ytom;
    size_t Zdim, Ztom;

	/** Is the volume previously masked?*/
	int boxsize;
    int nthrs;

    double sampling, lowRes, highRes, resStep;

    bool useHalves;

public:

    void defineParams();
    void readParams();
    void createSphere(MultidimArray<int> &maskNormalize, int halfboxsize);
    void extractSubtomos(const MultidimArray<double> &oddTomo, const MultidimArray<double> *evenTomo,
			MultidimArray<double> &subtomoOdd, MultidimArray<double> *subtomoEven,
			int halfboxsize, int xcoor, int ycoor, int zcoor, bool nextcoor);
    void setLocalResolutionSubtomo(const MultidimArray<double> &localResMap, MultidimArray<double> &tomo,
			int halfboxsize, int xcoor, int ycoor, int zcoor);
    void fillingBackground();
    void run();
};
//@}
#endif
