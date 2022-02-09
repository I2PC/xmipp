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
#ifndef _PROG_ELIMINATE_BY_ENERGY_HH
#define _PROG_ELIMINATE_BY_ENERGY_HH

#include "core/xmipp_metadata_program.h"

/**@defgroup EliminateByEnergyProgram Eliminate images whose energy is extremely large or extremely low
   @ingroup ReconsLibrary */
//@{
/// Threshold Parameters
class ProgEliminateByEnergy : public XmippMetadataProgram
{
public:
	// Confidence
	double confidence;
    // Reference variance
    double sigma20;
    // Min variance
    double minSigma2;
public:
    /** Read parameters from command line. */
    void readParams();

    /** Define Parameters */
    void defineParams();

    /** Show parameters */
    void show();

    /// Process image or volume
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

    /// Finish processing
    void finishProcessing();
};
//@}
#endif
