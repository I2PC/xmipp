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

#include "bspline_geo_transformer.h"
#include "data/filters.h"

template<typename T>
void BSplineGeoTransformer<T>::initialize(bool doAllocation) {
    const auto &s = this->getSettings();

    for (auto &hw : s.hw) {
        if ( ! dynamic_cast<CPU*>(hw)) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Instance of CPU is expected");
        }
    }
//    m_threadPool.resize(s.hw.size()); // FIXME DS set to requested number of thread
    m_threadPool.resize(CPU::findCores());

    if (doAllocation) {
        release();
        m_dest = std::unique_ptr<T[]>(new T[s.dims.size()]);
    }
}

template<typename T>
void BSplineGeoTransformer<T>::check() {
    const auto &s = this->getSettings();
    if (InterpolationDegree::Linear != s.degree) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Only linear interpolation is available");
    }
    if (s.doWrap) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Wrapping is not yet implemented");
    }
    if (InterpolationType::NToN != s.type) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Only NToN is currently implemented");
    }
}


template<typename T>
void BSplineGeoTransformer<T>::copySrcToDest() {
    bool isReady = this->isInitialized()
            && this->isSrcSet();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Instance is either not initialized or the 'src' has not been set.");
    }
    memcpy(m_dest.get(),
            m_src,
            this->getSettings().dims.size() * sizeof(T));
}

template<typename T>
void BSplineGeoTransformer<T>::release() {
    m_dest.release();
    setDefault();
}

template<typename T>
void BSplineGeoTransformer<T>::setDefault() {
    m_dest.reset();
    m_src = nullptr;
    m_threadPool.resize(1);
}

template<typename T>
bool BSplineGeoTransformer<T>::canBeReused(const BSplineTransformSettings<T> &s) const {
    bool result = true;
    if ( ! this->isInitialized()) {
        return false;
    }
    auto &sOrig = this->getSettings();
    result &= sOrig.dims.size() >= s.dims.size(); // previously, we needed more space
    result &= !(( ! sOrig.keepSrcCopy) && s.keepSrcCopy); // we have a problem if now we need to make a copy and before not

    return result;
}

template<typename T>
void BSplineGeoTransformer<T>::sum(T *dest, size_t firstN) {
    REPORT_ERROR(ERR_NOT_IMPLEMENTED, "This functionality is not yet available.");
}

template<typename T>
T *BSplineGeoTransformer<T>::interpolate(const std::vector<float> &matrices) {
    bool isReady = this->isInitialized()
            && this->isSrcSet();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Instance is either not initialized or the 'original' has not been set.");
    }
    const Dimensions dims = this->getSettings().dims;
    const size_t n = dims.n();
    const size_t z = dims.z();
    const size_t y = dims.y();
    const size_t x = dims.x();

    auto futures = std::vector<std::future<void>>();

    auto workload = [&](int id, size_t signalId){
        size_t offset = signalId * dims.sizeSingle();
        auto in = MultidimArray<T>(1, z, y, x, const_cast<T*>(m_src + offset)); // removing const, but data should not be changed
        auto out = MultidimArray<T>(1, z, y, x, m_dest.get() + offset);
        in.setXmippOrigin();
        out.setXmippOrigin();
        // compensate the movement
        Matrix2D<double> m(3,3);
        const float *f = matrices.data() + (9 * signalId);
        for (int i = 0; i < 9; ++i) {
            m.mdata[i] = f[i];
        }
        applyGeometry(LINEAR, out, in, m, true, DONT_WRAP);
    };

    for (size_t i = 0; i < n; ++i) {
        futures.emplace_back(m_threadPool.push(workload, i));
    }
    for (auto &f : futures) {
        f.get();
    }
    return m_dest.get();
}

// explicit instantiation
template class BSplineGeoTransformer<float>;
template class BSplineGeoTransformer<double>;
