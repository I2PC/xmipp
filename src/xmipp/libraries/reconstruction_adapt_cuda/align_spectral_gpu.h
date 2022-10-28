/***************************************************************************
 *
 * Authors:    Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

#ifndef LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_ALIGN_SPECTRAL_GPU_H_
#define LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_ALIGN_SPECTRAL_GPU_H_

#include "reconstruction/align_spectral.h"
#include "reconstruction_cuda/cuda_bspline_geo_transformer.h"

/**@defgroup ProgAlignSpectralGPU Align Spectral GPU
   @ingroup ReconsLibrary */
//@{
namespace Alignment {

template<typename T>
class ProgAlignSpectralGPU : public ProgAlignSpectral<T> {
public:

    void defineParams() override;
    void readParams() override;
    void show() const override;

private:
    std::vector<uint> m_devices;

};

} // namespace Alignment

#endif //LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_ALIGN_SPECTRAL_GPU_H_