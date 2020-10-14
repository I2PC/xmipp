/***************************************************************************
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
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

#include <core/xmipp_program.h>

/** Apply some filter operation on images, or selfiles */
class ProgImageResiduals: public XmippMetadataProgram
{
public:
    void defineParams();
    void readParams();
    void preProcess();
    void postProcess();
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

    // Normalize divergence
    bool normalizeDivergence;

    // Autocorrelation
    Matrix2D<double> R;
    Image<double> IR;

    // Mean and stddev of the residuals
    size_t i;
    MultidimArray<double> resmean, resvar;
}
;//end of class ProgFilter

/// Compute the divergence between two covariance matrices
double computeCovarianceMatrixDivergence(const Matrix2D<double> &C1, const Matrix2D<double> &C2);



