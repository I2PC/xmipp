/***************************************************************************
 *
 * Authors:    Estrella Fernandez Gimenez         me.fernandez@cnb.csic.es (2019)
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
#ifndef LIBRARIES_RECONSTRUCTION_VOLUME_SUBTRACTION_H_
#define LIBRARIES_RECONSTRUCTION_VOLUME_SUBTRACTION_H_

#include "core/xmipp_program.h"
#include <core/xmipp_fftw.h>
#include <data/fourier_filter.h>


class ProgVolumeSubtraction: public XmippProgram {
/*This class contains methods that are use to adjust an input volume (V) to a another reference volume (V1) through the use
 of Projectors Onto Convex Sets (POCS) and to perform the subtraction between them. Other methods contained in this class
 are used to pre-process and operate with the volumes*/
public:
  FileName fnVol2;
  FileName fnVol1;
  FileName fnOutVol;
  FileName fnMask1;
  FileName fnMask2;
  FileName fnMaskSub;
  FileName fnVol1F;
  FileName fnVol2A;

  bool computeE;
  size_t iter;

	/// Read arguments
	void readParams();

	/// Show
	void show() const;

	/// Define parameters
	void defineParams();

	/** Run */
	void run();

};
//@}
#endif
