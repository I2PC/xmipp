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
#ifndef _PROG_TOMO_MAP_BACK
#define _PROG_TOMO_MAP_BACK

#include "core/xmipp_image.h"
#include "core/xmipp_program.h"
#include "core/metadata.h"

///@defgroup TomoMapBack Tomo map back
///@ingroup ReconsLibrary
//@{
/** Map back parameters. */
class ProgTomoMapBack: public XmippProgram
{
public:
    /// Input
    FileName fn_tomo, fn_geom, fn_ref, fn_out;
    String modeStr;

public:
    // Input volume
    Image<double> tomo, reference;
    MetaData mdGeom;
    int mode;
    double K;
    double threshold;

public:
    /// Read arguments
    void readParams();

    /// Show
    void show() const;

    /// Define parameters
    void defineParams();

    /** Produce side info.*/
    void produce_side_info();

    /** Run */
    void run();
};
//@}
#endif
