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

#ifndef _PROG_CLASSIFICATION_FIRST_SPLIT3
#define _PROG_CLASSIFICATION_FIRST_SPLIT3

#include "core/metadata.h"
#include "core/xmipp_filename.h"
#include "core/xmipp_image.h"
#include "core/xmipp_program.h"
#include "data/mask.h"
#include "data/fourier_projection.h"

/**@defgroup ClassificationFirstSplit Classification first split2
   @ingroup ReconsLibrary */
//@{
/** Classification First Split 2 Parameters. */
class ProgClassifyFirstSplit3: public XmippProgram
{
public:
    /** Directional classes */
    FileName fnClasses;
    /** Rootname */
    FileName fnRoot;
    /** Number of iterations */
    int Niter;
    /** Number of samples per reconstruction */
    int Nsamples;
    /** Symmetry */
    FileName fnSym;
    /** External mask */
    bool externalMask;
    /** Mask */
    Mask mask;
    /** Learning rate */
    double alpha;
    /** String with MPI command */
    String mpiCommand;
    bool mpiUse;
public:
    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Define parameters
    void defineParams();

    /// Run
    void run();

    void updateVolume(const std::vector<size_t> &objIds1, const FileName &fnOut, FourierProjector &projector);

    void calculateProjectedIms (size_t id, double &corrI_P1, double &corrI_P2);
public:
    Image<double> V, imgV;
    MultidimArray<double> projV;
    int Nvols;
    size_t maskSize;
    double sumCorrDiff, sum2CorrDiff, sumCorr1, sumCorr2, sum2Corr1, sum2Corr2;
    int count1, count2, countChange, countTotal;
    MetaData md;
	Projection PV;
	FourierProjector *projectorV1, *projectorV2;
	int countSwap, countRandomSwap, countNormalSwap;
};
//@}
#endif
