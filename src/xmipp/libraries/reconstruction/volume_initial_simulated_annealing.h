/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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

#ifndef __VOLUME_INITIAL_H_H
#define __VOLUME_INITIAL_H_H

#include <core/xmipp_program.h>
#include <core/xmipp_threads.h>
#include "reconstruct_fourier.h"
#include "angular_project_library.h"

/**@defgroup VolumeInitialH Heuristic Initial Volume
   @ingroup ReconsLibrary */
//@{

/** Auxiliary class for aligning the input metadata */
class ThreadVolumeInitialAlignment
{
public:
	MetaData mdReconstruction;
	double sumCorr, sumImprovement;
};

class GalleryImage
{
public:
	FileName fnImg;
	double rot, tilt;
};

class InputImage
{
public:
	FileName fnImg;
	double maxcc;
};

/** Random reconstruction parameters. */
class ProgVolumeInitialSimulatedAnnealing: public XmippProgram
{
public:
    /** Filenames */
    FileName fnIn, fnRoot, fnSym, fnInit;

    /** First temperature */
    double T0;

    /** Total number of iterations */
    int NiterRandom, NiterGreedy;

    /** Rejection percentage */
    double rejection;

    /** Number of threads */
    int Nthr;

    /** Positive constraint */
    bool positiveConstraint;

    /** Keep intermediate volumes */
    bool keepIntermediateVolumes;

    /** Angular sampling */
    double angularSampling;

public: // Internal members
    MetaData mdIn, mdReconstruction;

    // Set of images in the gallery, it should be a metadata but metadatas do not support threads
    std::vector<GalleryImage> mdGallery;

    // Set of input images
    std::vector<InputImage> mdInp;

    // Filenames
    FileName fnAngles, fnVolume, fnGallery, fnGalleryMetaData;

    // Images
    Image<double> gallery, inputImages;

	// Current iteration
	int iter;

	// Array with information for the threads
	ThreadVolumeInitialAlignment *threadResults;

	// Mutex to update MaxCC
	Mutex mutexMaxCC;

	// Iteration statistics
	double sumCorr, sumImprovement;
public:
    /// Read arguments from command line
    void readParams();

    /// Read arguments from command line
    void defineParams();

    /** Show. */
    void show();

    /** Run. */
    void run();

    /// Produce side info: fill arrays with relevant transformation matrices
    void produceSideinfo();

    /// Reconstruct current volume
    void reconstructCurrent();

    /// Generate projections from the current volume
    void generateProjections();

    /// Filter by correlation
    void filterByCorrelation();
};
//@}
#endif
