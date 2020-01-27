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

#include "new_geo_transformer.h"

template<typename T>
void NewGeoTransformer<T>::init(bool doAllocation) {
    const auto &s = this->getSettings();

    for (auto &hw : s.hw) {
        if ( ! dynamic_cast<CPU*>(hw)) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Instance of CPU is expected");
        }
    }
    m_threadPool.resize(s.hw.size());

    if (doAllocation) {
        release();
        m_copy = std::unique_ptr<T[]>(new T[s.dims.size()]);
    }
}

template<typename T>
void NewGeoTransformer<T>::check() {
    const auto &s = this->getSettings();
    if (auto *method = dynamic_cast<BSplineInterpolation<T>*>(s.method)) {
        checkBSpline(method);
    } else {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Only BSpline interpolation is available");
    }
}

template<typename T>
void NewGeoTransformer<T>::checkBSpline(const BSplineInterpolation<T> *i) {
    const auto &s = this->getSettings();
    if (InterpolationDegree::Linear != s.degree) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Only linear interpolation is available");
    }
}

template<typename T>
void NewGeoTransformer<T>::copyOriginalToCopy() {
    bool isReady = this->isInitialized()
            && this->isOrigLoaded();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Instance is either not initialized or the 'original' has not been set.");
    }
    memcpy(m_copy.get(),
            m_orig,
            this->getSettings().dims.size() * sizeof(T));
}

template<typename T>
void NewGeoTransformer<T>::release() {
    m_copy.release();
}

template<typename T>
bool NewGeoTransformer<T>::canBeReused(const GeoTransformerSetting &s) const {
    bool result = true;
    if ( ! this->isInitialized()) {
        return false;
    }
    auto &sOrig = this->getSettings();
    result &= sOrig.dims.size() >= s.dims.size(); // previously, we needed more space
    result &= !(( ! sOrig.createReferenceCopy) && s.createReferenceCopy); // we have a problem if now we need to make a copy and before not
    result &= sOrig.type == s.type;

    return result;
}


template<typename T>
T *NewGeoTransformer<T>::interpolate(const std::vector<float> &matrices) {
    bool isReady = this->isInitialized()
            && this->isOrigLoaded();
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
        auto in = MultidimArray<T>(1, z, y, x, const_cast<T*>(m_orig + offset)); // removing const, but data should not be changed
        auto out = MultidimArray<T>(1, z, y, x, m_copy.get() + offset);
        in.setXmippOrigin();
        out.setXmippOrigin();
        // compensate the movement
        Matrix2D<double> m;
        m.initIdentity(3);
        const float *f = matrices.data() + (9 * signalId);
        for (int i = 0; i < 9; ++i) {
            m.mdata[i] = f[i];
        }
        applyGeometry(LINEAR, out, in, m, false, DONT_WRAP);
    };

    for (size_t i = 0; i < n; ++i) {
        futures.emplace_back(m_threadPool.push(workload, i));
    }
    for (auto &f : futures) {
        f.get();
    }
    return m_copy.get();
}

// explicit instantiation
template class NewGeoTransformer<float>;
template class NewGeoTransformer<double>;
