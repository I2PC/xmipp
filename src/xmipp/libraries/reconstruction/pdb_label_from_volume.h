/***************************************************************************
 *
 * Authors:
 *  Erney Ramirez-Aportela (eramirea@cnb.csic.es)
 *  Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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
#ifndef _PROG_PDB_FROM_VOLUME_HH
#define _PROG_PDB_FROM_VOLUME_HH

#include "core/xmipp_program.h"
#include "core/multidim_array.h"
#include "core/xmipp_filename.h"

/**@defgroup PDBPhantom convert_pdb2vol (PDB Phantom program)
   @ingroup ReconsLibrary */
//@{
/* PDB Phantom Program Parameters ------------------------------------------ */
/** Parameter class for the PDB Phantom program */
class ProgPdbValueToVol: public XmippProgram
{
public:
    /** Sampling rate */
    double Ts;

    /** Radius */
    double radius;

    // Origin
    StringVector origin;

    /** Origin by user */
    bool defOrig;

    /** Use Mask */
    bool withMask;

    /** PDB file */
    FileName fn_pdb;

    /** Volume file */
    FileName fnVol, fnMask;
	MultidimArray<double> inputVol, inputMask;

    /** Output fileroot */
    FileName fn_out, fnMD;


    /** Final size in pixels */
    int output_dim;
    

public:

    /** Params definitions */
    void defineParams();
    /** Read from a command line.
        An exception might be thrown by any of the internal conversions,
        this would mean that there is an error in the command line and you
        might show a usage message. */
    void readParams();

    /** Produce side information.
        Produce the atomic profiles. */
    void produceSideInfo();

    /** Show parameters. */
    void show();

    /** Run. */
    void run();
public:
    /* Downsampling factor */
    int M;

    // Protein geometry
    Matrix1D<double> centerOfMass, limit;

    /* Protein geometry */
    void computeProteinGeometry();

};
//@}
#endif
