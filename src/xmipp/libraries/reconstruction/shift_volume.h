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
#ifndef _PROG_SHIFT_VOLUME
#define _PROG_SHIFT_VOLUME

#include "core/xmipp_image.h"
#include "core/xmipp_program.h"
#include "core/metadata_vec.h"

/** Shift volume parameters. */
class ProgShiftVolume: public XmippProgram
{
private:
    /// Input volume
    FileName fn_vol;
    Image<double> vol;
    // Output (shifted) volume
    FileName fn_out;
    Image<double> outVol;
    // Coordinates
    int x;
    int y;
    int z;

private:
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
