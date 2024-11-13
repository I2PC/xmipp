/***************************************************************************
 * Authors:     AUTHOR_NAME (jvargas@cnb.csic.es)
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

#ifndef IMAGE_ADJUST_LSQ_H_
#define IMAGE_ADJUST_LSQ_H_

#include "core/xmipp_metadata_program.h"
#include "data/mask.h"
#include "data/fourier_projection.h"

#

class ProgImageGrayAdjustLsq: public XmippMetadataProgram
{


public:
    FileName referenceFilename;

public:

    void readParams();

    void defineParams();

public:
    void preProcess();

    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

protected:
    Image<double> m_image;
    Image<double> m_volume;
    std::unique_ptr<FourierProjector> m_projector;

    bool fitLeastSquares(const MultidimArray<double> &ref, const MultidimArray<double> &exp, double &a, double &b);
};

#endif /* IMAGE_ADJUST_LSQ_H_ */
