/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 * 			   Carlos Oscar S. Sorzano            coss@cnb.csic.es (2019)
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

#ifndef _PROG_SPLIT_ODD_EVEN
#define _PROG_SPLIT_ODD_EVEN

#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <string>
#include "core/metadata_extension.h"

/**@defgroup Odd Even
   @ingroup ReconsLibrary */
//@{
/** SSNR parameters. */

class ProgOddEven : public XmippProgram
{
private:
	 /** Filenames */
	FileName fnOut_odd, fnOut_even, fnImg, splitType;
	bool sumFrames;

private:
    void defineParams();
    void readParams();
    // This function generates the metadata associated to the input image stack
    void fromimageToMd(FileName fnImg, MetaData &movienew, size_t &Xdim, size_t &Ydim);
    void run();


};
//@}
#endif

