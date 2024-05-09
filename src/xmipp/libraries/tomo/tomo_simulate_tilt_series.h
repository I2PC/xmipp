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

#ifndef _PROG_TOMO_SIMULATE_TILT_SERIES
#define _PROG_TOMO_SIMULATE_TILT_SERIES

#include <complex>
#include "core/xmipp_program.h"
#include "core/xmipp_filename.h"
#include "core/metadata_vec.h"

template<typename T>
class MultidimArray;

/** Movie alignment correlation Parameters. */
class ProgTomoSimulateTiltseries: public XmippProgram
{
public:
    /** input filename of the metadata with coordinates */
    FileName fnCoords;

    /** input filename with a volume of the protein*/
    FileName fnVol;

    FileName fnFid;

    /** Tilting parameters and std of noise*/
    double maxTilt, minTilt, tiltStep, sigmaNoise;

    /** Fiducial size */
    double fidDiameter;

    /** output tilt series */
    FileName fnTsOut, fnTomoOut;

	/** dimensions of the tilt series*/
	int xdim;
	int ydim;

	/** number of fiducials */
	int nfids;

	/** Tomogram thickness */
	int thickness;

    /** Sampling rate */
    double sampling;

public:
    /// Read argument from command line
    void readParams();

    /// create fiducial
    void createFiducial(MultidimArray<double> &fidImage, int boxsize);

    /// create a cirte to limit the projection extension
    void createSphere(MultidimArray<int> &mask, int boxsize);

    void maskingRotatedSubtomo(MultidimArray<double> &subtomo, int boxsize);

    void placeSubtomoInTomo(const MultidimArray<double> &subtomo, MultidimArray<double> &tomo,
			const int xcoord, const int ycoord, const int zcood, const size_t halfboxsize);

    void createFiducial(MultidimArray<double> &fidImage, MultidimArray<double> &fidVol, int fidSize);

    /// Define parameters
    void defineParams();

    /// Run
    void run();

};
//@}
#endif
