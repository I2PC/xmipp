/***************************************************************************
 *
 * Authors:    Estrella Fernandez Gimenez         me.fernandez@cnb.csic.es (2023)
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
#ifndef LIBRARIES_RECONSTRUCTION_LOCAL_VOLUME_ADJUST_H_
#define LIBRARIES_RECONSTRUCTION_LOCAL_VOLUME_ADJUST_H_

#include "core/xmipp_program.h"
#include <core/xmipp_fftw.h>
#include <data/fourier_filter.h>


class ProgLocalVolumeAdjust: public XmippProgram {
/*This class contains methods that are use to adjust an input volume (V) to a another reference volume (V1) locally
by using a least squares solution in a sliding window*/

private:
  FileName fnVol2;
  FileName fnVol1;
  FileName fnOutVol;
  FileName fnMask;
  bool performSubtraction;
  int neighborhood;

	/// Read arguments
	void readParams() override;

	/// Show
	void show() const override;

	/// Define parameters
	void defineParams() override;

	/** Run */
	void run() override;

};
//@}
#endif
