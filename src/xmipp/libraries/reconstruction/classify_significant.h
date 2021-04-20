/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano coss@cnb.csic.es
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

#ifndef _PROG_CLASSIFY_SIGNIFICANT
#define _PROG_CLASSIFY_SIGNIFICANT

#include "core/xmipp_program.h"
#include "data/fourier_projection.h"
#include "core/metadata.h"

/**@defgroup ClassifySignificant Classify a set of images into a discrete set of classes
   @ingroup ReconsLibrary */
//@{

/** Classify Significant Parameters. */
class ProgClassifySignificant: public XmippProgram
{
public:
    /** Filename of the reference volumes */
    FileName fnVols;
    /** Filename of indexes to study */
    FileName fnIds;
    /** Filename of angles assigned */
    FileName fnAngles;
    /** Output file */
    FileName fnOut;
    /** FSC file */
    int numFsc;
    //FileName fnFsc1, fnFsc2;
    /** Padding factor */
    int pad;
    /** Min. Weight */
    double wmin;
    /** Flag to select only the images belonging only to the set intersection */
    bool onlyIntersection;
    /** Minimum number of votes to consider an image belonging to a volume */
    int numVotes;
    /** To check if there is FSC provided by user */
    bool isFsc;

public:
    // Fourier projector
    std::vector<FourierProjector *> projector;
    // Set of FSCs
    std::vector<FileName> setFsc;
	// Set of Ids
	std::vector<size_t> setIds;
	// Set of Angles
	std::vector<VMetaData> setAngles;
	// Set of Angles
	std::vector<VMetaData> classifiedAngles;
	// Current row
	std::vector<size_t> currentRowIdx;
	// Set of Angles for a particular image
	std::vector<VMetaData> subsetAngles;
	// Set of projections for a particular image, they act as a pool of projections
	std::vector<MultidimArray<double> *> subsetProjections;
	// Set of indexes of the projections for a particular image
	std::vector< std::vector<size_t> > subsetProjectionIdx;
	// Experimental image
	std::vector<Image<double> *> Iexp;
	// Projection aux
	Projection Paux;
	// FSC values
	std::vector< std::vector<double> > setFscValues;
public:
    /// Destructor
    ~ProgClassifySignificant();

    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Define parameters
    void defineParams();

    /** Produce side info.
        An exception is thrown if any of the files is not found*/
    void produceSideInfo();

    /** Predict angles and shift.
        At the input the pose parameters must have an initial guess of the
        parameters. At the output they have the estimated pose.*/
    void run();

    /** Generate the projection of a given volume following the instructions of currentRow.
     * The result is stored in subsetProjections at poolIdx
     */
    void generateProjection(size_t volumeIdx, size_t poolIdx, MDRow &currentRow);

    /** Choose the subset for particleID and generate its projections */
    void selectSubset(size_t particleId, bool &flagEmpty);

    /** Update class */
    void updateClass(int n, double wn);
};
//@}
#endif
