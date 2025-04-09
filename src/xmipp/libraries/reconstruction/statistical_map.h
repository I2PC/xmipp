/***************************************************************************
 *
 * Authors:    Federico P. de Isidro-Gomez (federico.pdeisidro@astx.com)
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

#ifndef _PROG_STATISTICAL_MAP
#define _PROG_STATISTICAL_MAP

#include "core/metadata_vec.h"
// #include "core/xmipp_program.h"
// #include "core/xmipp_image.h"
// #include "data/fourier_filter.h"
// #include "data/fourier_projection.h"
// #include "core/xmipp_metadata_program.h"

#define VERBOSE_OUTPUT
#define DEBUG_OUTPUT_FILES

/**@defgroup ProgStatisticalMap Calculates statistical map
   @ingroup ReconsLibrary */
//@{
/** Calculate statistical map from a pool of input maps */

class ProgStatisticalMap: public XmippProgram
{
 public:
    // Input params
    FileName fn_in;     // Input metadata with volume pool
    FileName fn_oroot;  // Output location for saving statistical maps

    // Side info variables
    FileName fn_out_avg_map;
    FileName fn_out_std_map;

    // Volume dimensions
    size_t Xdim;
    size_t Ydim;
    size_t Zdim;
    size_t Ndim;

    // Data variables
    FileName fn_V;              // Filename for each input volume from pool
 	Image<double> V;            // Each input volume from pool
 	Image<double> avgVolume;    // Average volume
 	Image<double> stdVolume;    // Standard deviation volume

    // Particle metadata
    MetaDataVec mapPoolMD;
    MDRowVec row;

public:

    // ---------------------- IN/OUT METHODS -----------------------------
    // Define parameters
    void defineParams() override;
    // Read argument
    void readParams() override;
    // Show
    void show() const override;
    // Read input map from pool
    void readVolume(const MDRow &rowIn);
    // Write output statistical map
    void writeStatisticalMap(MDRow &rowOut, FileName, Image<double> &, double, double, double);

    // ----------------------- MAIN METHODS ------------------------------
    void run();

    // ----------------------- CORE METHODS ------------------------------
    void processVolume(FileName fn_vol);

    // ---------------------- UTILS METHODS ------------------------------
    // Generate side info
    void generateSideInfo();

};
//@}
#endif