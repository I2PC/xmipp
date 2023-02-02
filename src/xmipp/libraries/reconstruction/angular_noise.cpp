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

#include "angular_noise.h"

#include "core/matrix1d.h"
#include "core/transformations.h"

#include <cmath>

// Empty constructor =======================================================
ProgAngularNoise::ProgAngularNoise()
    : rd()
    , randomNorm(0.0, 1.0)
    , randomAngle(0.0, 2.0 * M_PI)
{
    produces_a_metadata = true;
    produces_an_output = true;
}

// Read arguments ==========================================================
void ProgAngularNoise::readParams()
{
    XmippMetadataProgram::readParams();
    sigmaRotation = getDoubleParam("--sigmaRotation");
    sigmaShift = getDoubleParam("--sigmaShift");
}

// Show ====================================================================
void ProgAngularNoise::show()
{
    if (!verbose)
        return;
    XmippMetadataProgram::show();
    //TODO
}

// usage ===================================================================
void ProgAngularNoise::defineParams()
{
    addUsageLine("Add pose assignment noise to a metadata file");
    XmippMetadataProgram::defineParams();
    addParamsLine("   --sigmaRotation <sigma=1>             : Standard deviation of noise to be added to the rotation. In degrees");
    addParamsLine("   --sigmaShift <sigma=1>                : Standard deviation of noise to be added to the shift assignment. In pixels");
}

// Run ---------------------------------------------------------------------
// Predict shift and psi ---------------------------------------------------
// #define DEBUG
void ProgAngularNoise::processImage(const FileName&, const FileName&, const MDRow &rowIn, MDRow &rowOut)
{
    // Get the input transformation
    geo2TransformationMatrix(rowIn, transformMatrix);
    
    // Apply a random shift
    if (sigmaShift > 0.0)
    {
        Matrix1D<double> shift(2);
        getRandomUnitVector(shift);

        std::normal_distribution<double> magDist(0.0, sigmaShift);
        const auto mag = magDist(rd);

        shift *= mag;
        dMij(transformMatrix, 0, 3) = XX(shift);
        dMij(transformMatrix, 1, 3) = YY(shift);
    }

    // Apply random rotation
    if (sigmaRotation > 0.0) 
    {
        // Select a random axis:
        Matrix1D<double> axis(3);
        getRandomUnitVector(axis);

        // Select the angle
        std::normal_distribution<double> angleDist(0.0, sigmaRotation);
        const auto angle = angleDist(rd);

        // Get the rotation matrix
        rotation3DMatrix(angle, axis, rotationMatrix, true);
        
        // Apply transform
        transformMatrix = rotationMatrix * transformMatrix;
    }

    // Set the output transform
    transformationMatrix2Geo(transformMatrix, rowOut);
}

// Finish processing ---------------------------------------------------------
//void ProgAngularNoise::postProcess()
//{
//    mdOut.write(fn_out);
//}

void ProgAngularNoise::getRandomUnitVector(Matrix1D<double>& result)
{
    if(VEC_XSIZE(result) == 2) 
    {
        const auto phi = randomAngle(rd);
        const auto cosPhi = std::cos(phi);
        const auto sinPhi = std::sin(phi);
        
        XX(result) = cosPhi;
        YY(result) = sinPhi;

    } 
    else if (VEC_XSIZE(result) == 3)
    {
        // Based on:
        // https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
        const auto lambda = std::acos(2*randomNorm(rd) - 1) - (M_PI/2);
        const auto phi = randomAngle(rd);
        const auto cosLambda = std::cos(lambda);
        const auto sinLambda = std::sin(lambda);
        const auto cosPhi = std::cos(phi);
        const auto sinPhi = std::sin(phi);

        XX(result) = cosLambda*cosPhi;
        YY(result) = cosLambda*sinPhi;
        ZZ(result) = sinLambda;
    } 
    else
    {
        //NOT IMPLEMENTED
    }
}