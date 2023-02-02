/***************************************************************************
 *
 * Authors: Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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
#ifndef _PROG_ANGULAR_PREDICT
#define _PROG_ANGULAR_PREDICT

#include "core/xmipp_metadata_program.h"
#include "core/matrix1d.h"
#include "core/multidim_array.h"
#include "core/metadata_vec.h"
#include "angular_distance.h"

#include <random>

/**@defgroup AngularNoise
   @ingroup ReconsLibrary */
//@{
/** Add angular noise. */
class ProgAngularNoise: public XmippMetadataProgram
{
public:
    /// Sigma in rotation
    double sigmaRotation;
    /// Sigma in shift
    double sigmaShift;

public:
    ProgAngularNoise();

    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Usage
    void defineParams();

    /** Add noise*/
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

//    void postProcess();

private:
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 generator; // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> randomNorm;
    std::uniform_real_distribution<double> randomAngle;
    Matrix2D<double> rotationMatrix;
    Matrix2D<double> transformMatrix;

    void getRandomUnitVector(Matrix1D<double>& result);

};
//@}
#endif
