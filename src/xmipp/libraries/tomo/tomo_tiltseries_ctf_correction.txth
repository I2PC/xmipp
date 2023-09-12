/***************************************************************************
 *
 * Authors:    J.L. Vilas jlvilas@cnb.csic.es           
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
/* This program is strongly based on the unblur code
 * (http://grigoriefflab.janelia.org/unblur)
which has been realeased under the following copyright:

The Janelia Farm Research Campus Software Copyright 1.1

Copyright (c) 2014, Howard Hughes Medical Institute, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:


Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
Neither the name of the Howard Hughes Medical Institute nor the names of
its contributors may be used to endorse or promote products derived from
this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
REASONABLE ROYALTIES; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/
#ifndef _PROG_TOMO_TILTSERIES_CTF_CORRECTION
#define _PROG_TOMO_TILTSERIES_CTF_CORRECTION

#include "core/xmipp_metadata_program.h"
#include "data/ctf.h"
#include "data/wiener2d.h"
#include "core/xmipp_image.h"
#include "data/filters.h"

/**@defgroup Correct CTF by Wiener filter in 2D
   @ingroup ReconsLibrary */
//@{
class ProgTomoTSCTFCorrection: public XmippMetadataProgram
{


public:
    /* Wiener class*/
    Wiener2D WF;

    bool phase_flipped;

    /** Padding factor */
    double	pad;

    bool isIsotropic;

    bool correct_envelope;

    /// Wiener filter constant
    double wiener_constant;

    /// Sampling rate
    double sampling_rate;

private:

    /// Input filename
    FileName fnTs;

    /// Output filename or folder
    FileName fnOut;

    /// sampling rate
    double sampling;

public:

    void readParams();

    void defineParams();

public:

    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

    void generateWienerFilter(MultidimArray<double> &Mwien, CTFDescription &ctf);

	void run();

	void show();

public:
	Image<double> img;

	CTFDescription ctf;

	size_t Ydim, Xdim;

	MultidimArray<double> Mwien;
	MultidimArray<std::complex<double> > Faux;
    FourierTransformer transformer;
};




#endif
