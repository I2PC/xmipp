/***************************************************************************
 * Authors:     AUTHOR_NAME (eramirez@cnb.csic.es)
 *
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

#ifndef PROT_TRANSF_VOL
#define PROT_TRANSF_VOL

#include <iostream>
#include <math.h>
#include <complex>
#include <core/xmipp_image.h>
#include <core/xmipp_program.h>
#include <core/alglib/ap.h>




class ProgTransFromVol: public XmippMetadataProgram
{
public:
    /** Filenames */
    FileName fnVol, outVol, fnMdIn, fnMask;
	bool initvol;

    /** Particle size*/
    int boxSize;

public:
    Matrix1D<double> center, projectedCenter;
	Matrix2D<double> A;
	Image<double> Iin, Iout;
    size_t objId;


public:
	ProgTransFromVol();
    void defineParams();
    void readParams();
    void show();
    void startProcessing();
    void preProcess();
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);
};
#endif

