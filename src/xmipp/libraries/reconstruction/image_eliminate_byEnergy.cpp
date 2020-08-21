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

#include "image_eliminate_byEnergy.h"
#include "core/metadata.h"
#include "core/xmipp_image.h"

/* Read parameters --------------------------------------------------------- */
void ProgEliminateByEnergy::readParams()
{
    XmippMetadataProgram::readParams();
    confidence=getDoubleParam("--confidence");
    sigma20=getDoubleParam("--sigma2");
    minSigma2=getDoubleParam("--minSigma2");
}

/* Usage ------------------------------------------------------------------- */
void ProgEliminateByEnergy::defineParams()
{
    addUsageLine("Eliminate images whose variance is extremely large");
    each_image_produces_an_output=false;
    produces_an_output=true;
    produces_a_metadata=true;
    XmippMetadataProgram::defineParams();
    addParamsLine("[--confidence <conf=0.99>] : Remove an image if its variance is outside this confidence beyond sigma^2_0");
    addParamsLine("[--sigma2 <sigma20=1>] : Reference variance");
    addParamsLine("[--minSigma2 <sigma2=0.01>] : Minimum variance");
}

/* Show ------------------------------------------------------------------- */
void ProgEliminateByEnergy::show()
{
    if (verbose==0)
        return;
    XmippMetadataProgram::show();
    std::cout
    << "Confidence:    " << confidence    << std::endl
    << "Sigma2:        " << sigma20       << std::endl
    << "MinSigma2:     " << minSigma2     << std::endl;
}

/* Process image ------------------------------------------------------------- */
void ProgEliminateByEnergy::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
    rowOut = rowIn;
    Image<double> I;
    I.read(fnImg);
    double avg, stddev;
    I().computeAvgStdev(avg,stddev);
    double sigma2=stddev*stddev;

    double z=(sigma2/sigma20-1);
    double zalpha=fabs(icdf_gauss(confidence));
    if (z>zalpha || sigma2<minSigma2 || !std::isfinite(stddev) || fabs(avg)>sigma20/9.0)
        rowOut.setValue(MDL_ENABLED,-1);
    else
        rowOut.setValue(MDL_ENABLED,1);
}

void ProgEliminateByEnergy::finishProcessing() {
    getOutputMd()->removeDisabled();
    XmippMetadataProgram::finishProcessing();
}
