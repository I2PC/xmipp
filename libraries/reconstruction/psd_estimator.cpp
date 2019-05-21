/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
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

#include "psd_estimator.h"

template<typename T>
std::vector<Rectangle<Point2D<size_t>>> PSDEstimator<T>::getPatchesLocation(
        const std::pair<size_t, size_t> &borders,
        const Dimensions &micrograph,
        const Dimensions &patch,
        float overlap) {
    assert(micrograph.x() >= patch.x());
    assert(micrograph.y() >= patch.y());
    assert((2 * borders.first + patch.x()) <= micrograph.x());
    assert((2 * borders.second + patch.y()) <= micrograph.y());
    assert(overlap >= 0.f);
    assert(overlap < 1.f);

    size_t stepX = std::max((1 - overlap) * patch.x(), 1.f);
    size_t stepY = std::max((1 - overlap) * patch.y(), 1.f);

    size_t maxX = micrograph.x() - borders.first - patch.x();
    size_t maxY = micrograph.y() - borders.second - patch.y();

    auto result = std::vector<Rectangle<Point2D<size_t>>>();

    size_t y = borders.second;
    while (y < (maxY + stepY)) {
        size_t ys = std::min(y, maxY);
        size_t ye = ys + patch.y() - 1;
        size_t x = borders.first;
        while (x < (maxX + stepX)) {
            size_t xs = std::min(x, maxX);
            size_t xe = xs + patch.x() - 1;
            auto tl = Point2D<size_t>(xs, ys);
            auto br = Point2D<size_t>(xe, ye);
            result.emplace_back(tl, br);
            x += stepX;
        }
        y += stepY;
    }
    return result;
}

template<typename T>
void PSDEstimator<T>::estimatePSD(const MultidimArray<T> &micrograph,
        float overlap, const Dimensions &patchDim, MultidimArray<T> &psd,
        unsigned fftThreads) {
    using transformer = FFTwT<T>;
    // get patch positions
    auto patches = getPatchesLocation({0, 0},
            Dimensions(micrograph.xdim, micrograph.ydim),
            patchDim,
            overlap);

    auto settings = FFTSettingsNew<T>(patchDim);

    // prepare data for FT - set proper sizes and allocate aligned dat for faster execution
    // XXX HACK
    MultidimArray<T> patchData;
    patchData.destroyData = false;
    patchData.setDimensions(patchDim.x(), patchDim.y(), 1, 1);
    patchData.nzyxdimAlloc = patchData.nzyxdim;
    patchData.data = (T*)transformer::allocateAligned(settings.sBytesSingle());

    MultidimArray<T> smoother;
    ProgCTFEstimateFromMicrograph::constructPieceSmoother(patchData, smoother);
    smoother.resetOrigin();

    auto patchFS = (std::complex<T>*)transformer::allocateAligned(settings.fBytesBatch());
    auto magnitudes = new T[settings.fElemsBatch()](); // initialize to zero

    auto hw = CPU(fftThreads);
    auto plan = transformer::createPlan(hw, settings, true);

    for (auto &p : patches) {
        // get patch data
        window2D(micrograph, patchData,
                p.tl.y, p.tl.x, p.br.y, p.br.x);
        // normalize, otherwise we would get 'white cross'
        patchData.statisticsAdjust((T)0, (T)1);
        patchData.resetOrigin();
        // apply edge attenuation
        patchData *= smoother;
        // perform FFT
        transformer::fft(plan, patchData.data, patchFS);
        // get average of amplitudes
        for (size_t n = 0; n < settings.fElemsBatch(); ++n) {
            auto v = patchFS[n]; // / (T)settings.sDim().xyz();
            auto mag = sqrt((v.real() * v.real()) + (v.imag() * v.imag()));
            magnitudes[n] += mag;
        }
    }

    // create other half
    psd.resizeNoCopy(patchData);
    half2whole(magnitudes, psd.data, settings, [&](bool mirror, T val){return val;});

    delete[] magnitudes;
    transformer::release(patchFS);
    transformer::release(plan);
    transformer::release(patchData.data);
}

// explicit instantiation
template class PSDEstimator<float>;
template class PSDEstimator<double>;
