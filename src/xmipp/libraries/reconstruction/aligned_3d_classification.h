/***************************************************************************
 * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

#ifndef ALIGNED_3D_CLASSIFICATION_H_
#define ALIGNED_3D_CLASSIFICATION_H_

#include "core/xmipp_metadata_program.h"
#include "core/xmipp_image.h"
#include "core/xmipp_filename.h"
#include "core/metadata_vec.h"
#include "data/mask.h"
#include "data/ctf.h"
#include "data/fourier_projection.h"

#include <vector>
#include <memory>

class ProgAligned3dClassification: public XmippMetadataProgram
{
public:
    void readParams() override;
    void defineParams() override;
    void preProcess() override;
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut) override;

public:
    bool useCtf;
    double padding;
    double maxFreq;
    FileName referenceVolumesMdFilename;

    MetaDataVec referenceVolumesMd;
    Mask mask;
    CTFDescription ctfDescription;

    std::vector<Image<double>> referenceVolumes;
    std::vector<FourierProjector> projectors;
    std::unique_ptr<FourierProjector> maskProjector;

    Image<double> inputImage;
    MultidimArray<double> ctfImage;

private:
    void readVolumes();
    void createProjectors();
    void createMaskProjector();

};

#endif /* ALIGNED_3D_CLASSIFICATION_H_ */
