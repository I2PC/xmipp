/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 * 			   Carlos Oscar S. Sorzano                   coss@cnb.csic.es
 * 			   Federico P. de Isidro Gómez		  fp.deisidro@cnb.csic.es
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
    FileName fnTom;
    FileName fnCoor;

    size_t Xdim, Xtom;
    size_t Ydim, Ytom;
    size_t Zdim, Ztom;

	/** Is the volume previously masked?*/
	int boxsize;
    int nthrs;

public:

    void defineParams();
    void readParams();
    void createSphere(MultidimArray<double> &maskNormalize, int halfboxsize);
    void extractSubtomo(MultidimArray<double> &subtomo, int halfboxsize, const MultidimArray<double> &tom,
						int xcoor, int ycoor, int zcoor, double &invertSign, bool &nextcoor);
    void run();
};
//@}
#endif
