/***************************************************************************
 *
 * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

#include "transform_band_map.h"

#include <core/xmipp_image.h>

void ProgTransformBandMap::defineParams() {
    addUsageLine("Compute the basis for each band of a set of images");

    addParamsLine("   -o <image_file>                 : Output image file for the band map");
    addParamsLine("   --bands <n_bands>               : Band count");
    addParamsLine("   --size <x> <y> <z>              : Band map size");
    addParamsLine("   --lowRes <digital_freq>         : Low resolution limit");
    addParamsLine("   --highRes <digital_freq>        : High resolution limit");
}

void ProgTransformBandMap::readParams() {
    fnOutput = getParam("-o");
    nBands = getIntParam("--bands");
    imageSizeX = getIntParam("--size", 0);
    imageSizeY = getIntParam("--size", 1);
    imageSizeZ = getIntParam("--size", 2);
    lowResLimit = getDoubleParam("--lowRes");
    highResLimit = getDoubleParam("--highRes");
}

void ProgTransformBandMap::show() const {
    if (verbose < 1) return;

    std::cout << "Output image                : " << fnOutput << "\n";
    std::cout << "Band count                  : " << nBands << "\n";
    std::cout << "Image size                  : " << imageSizeX << "x" << imageSizeY << "x" << imageSizeZ << "\n";
    std::cout << "Low resolution limit        : " << lowResLimit << "\n";
    std::cout << "High resolution limit       : " << highResLimit << "\n";
    
    std::cout.flush();
}

void ProgTransformBandMap::run() {
    // Compute the bands
    const auto frequencies = computeArithmeticBandFrecuencies(
        lowResLimit, 
        highResLimit,
        nBands
    );
    const auto bands = computeBands(
        imageSizeX, imageSizeY, imageSizeZ,
        frequencies 
    );

    // Show info
    /*const auto& sizes = m_bandMap.getBandSizes();
    assert(frequencies.size() == sizes.size()+1);
    std::cout << "Bands frequencies:\n";
    for(size_t i = 0; i < sizes.size(); ++i) {
        std::cout << "\t- Band " << i << ": ";
        std::cout << frequencies[i] << " - " << frequencies[i+1] << "rad ";
        std::cout << "(" << sizes[i] << " coeffs)\n"; 
    }*/

    // Write the map to disk
    Image<int>(bands).write(fnOutput);
}


std::vector<double> ProgTransformBandMap::computeArithmeticBandFrecuencies( double lowResLimit,
                                                                            double highResLimit,
                                                                            size_t nBands )
{
    std::vector<double> result;
    result.reserve(nBands+1);

    const auto delta = highResLimit - lowResLimit;
    const auto step = delta / nBands;
    for(size_t i = 0; i <= nBands; ++i) {
        result.push_back(lowResLimit + i*step);
    }

    return result;
}

std::vector<double> ProgTransformBandMap::computeGeometricBandFrecuencies(  double lowResLimit,
                                                                            double highResLimit,
                                                                            size_t nBands )
{
    std::vector<double> result;
    result.reserve(nBands+1);

    const auto ratio = highResLimit / lowResLimit;
    const auto step = std::pow(ratio, 1.0/nBands);
    result.push_back(lowResLimit);
    for(size_t i = 1; i <= nBands; ++i) {
        result.push_back(result.back()*step);
    }

    return result;
}

MultidimArray<int> ProgTransformBandMap::computeBands(  const size_t nx, 
                                                        const size_t ny, 
                                                        const size_t nz,
                                                        const std::vector<double>& frecuencies )
{
    // Determine the band in which falls each frequency
    MultidimArray<int> bands(nz, ny, nx/2 + 1); // Half size because of the FT's symmetry
    const auto zStep = 1.0 / nz;
    const auto yStep = 1.0 / ny;
    const auto xStep = 1.0 / nx;
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(bands) {
        // Determine the angular frequency
        auto fz = zStep*k;
        auto fy = yStep*i;
        auto fx = xStep*j;
        fz = (fz > 0.5) ? fz - 1.0 : fz;
        fy = (fy > 0.5) ? fy - 1.0 : fy;
        fx = (fx > 0.5) ? fx - 1.0 : fx;
        auto f = std::sqrt(fx*fx + fy*fy + fz*fz);

        // Determine the band in which belongs this frequency
        int band = 0;
        while(band < frecuencies.size() && f > frecuencies[band]) {
            ++band;
        }
        if(band >= frecuencies.size()) {
            band = -1; // Larger than the last threshold
        } else {
            --band; // Undo the last increment
        }

        DIRECT_A3D_ELEM(bands, k, i, j) = band;
    }

    return bands;
}

