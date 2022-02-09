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

#include "single_extrema_finder.h"

namespace ExtremaFinder {

template<typename T>
void SingleExtremaFinder<T>::setDefault() {
    m_cpu = nullptr;
}

template<typename T>
void SingleExtremaFinder<T>::release() {
    setDefault();
}

template<typename T>
void SingleExtremaFinder<T>::check() const {
    if (this->getSettings().hw.size() > 1) {
        std::cerr << "Search using multiple threads is not yet implemented. Single thread will be used.\n";
    }
}

template<typename T>
void SingleExtremaFinder<T>::initMax() {
    return initBasic();
}

template<typename T>
void SingleExtremaFinder<T>::findMax(const T *__restrict__ data) {
    auto kernel = [&](const T *d) {
        sFindMax(*m_cpu, this->getSettings().dims, d,
            this->getPositions().data(),
            this->getValues().data());
    };
    return findBasic(data, kernel);
}

template<typename T>
bool SingleExtremaFinder<T>::canBeReusedMax(const ExtremaFinderSettings &s) const {
    return true;
}

template<typename T>
void SingleExtremaFinder<T>::initLowest() {
    return initBasic();
}

template<typename T>
void SingleExtremaFinder<T>::findLowest(const T *__restrict__ data) {
    auto kernel = [&](const T *d) {
        sFindLowest(*m_cpu, this->getSettings().dims, d,
            this->getPositions().data(),
            this->getValues().data());
    };
    return findBasic(data, kernel);
}

template<typename T>
bool SingleExtremaFinder<T>::canBeReusedLowest(const ExtremaFinderSettings &s) const {
    return true;
}

template<typename T>
void SingleExtremaFinder<T>::initMaxAroundCenter() {
    return initBasic();
}

template<typename T>
void SingleExtremaFinder<T>::findMaxAroundCenter(const T *__restrict__ data) {
    auto s = this->getSettings();
    auto kernel2D = [&](const T *d) {
        sFindMax2DAroundCenter(*m_cpu, s.dims, d,
            this->getPositions().data(),
            this->getValues().data(),
            s.maxDistFromCenter);
    };
    if (s.dims.is2D()) {
        return findBasic(data, kernel2D);
    }
    REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
}

template<typename T>
bool SingleExtremaFinder<T>::canBeReusedMaxAroundCenter(const ExtremaFinderSettings &s) const {
    return true;
}

template<typename T>
void SingleExtremaFinder<T>::initLowestAroundCenter() {
    return initBasic();
}

template<typename T>
void SingleExtremaFinder<T>::findLowestAroundCenter(const T *__restrict__ data) {
    auto s = this->getSettings();
    auto kernel2D = [&](const T *d) {
        sFindLowest2DAroundCenter(*m_cpu, s.dims, d,
            this->getPositions().data(),
            this->getValues().data(),
            s.maxDistFromCenter);
    };
    if (s.dims.is2D()) {
        return findBasic(data, kernel2D);
    }
    REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
}

template<typename T>
bool SingleExtremaFinder<T>::canBeReusedLowestAroundCenter(const ExtremaFinderSettings &s) const {
    return true;
}


template<typename T>
void SingleExtremaFinder<T>::initBasic() {
    release();
    auto s = this->getSettings();
    if (0 == s.hw.size()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "At least one CPU thread is needed");
    }
    try {
        m_cpu = dynamic_cast<CPU*>(s.hw.at(0));
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of CPU expected");
    }
}

template<typename T>
template<typename KERNEL>
void SingleExtremaFinder<T>::findBasic(const T *__restrict__ data, const KERNEL &k) {
    bool isReady = this->isInitialized();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() first");
    }
    k(data);
}

template<typename T>
void SingleExtremaFinder<T>::sFindUniversalChecks(
        const Dimensions &dims,
        const T *__restrict__ data,
        float *__restrict__ positions,
        T *__restrict__ values) {
    // check input
    assert(dims.sizeSingle() > 0);
    assert(dims.n() > 0);
    assert(nullptr != data);
    assert((nullptr != positions) || (nullptr != values));
}

template<typename T>
void SingleExtremaFinder<T>::sFindMax(const CPU &cpu,
    const Dimensions &dims,
    const T *__restrict__ data,
    float *__restrict__ positions,
    T *__restrict__ values) {
    sFindUniversalChecks(dims, data, positions, values);

    if (dims.isPadded()) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
    } else {
        // locate max
        for (size_t n = 0; n < dims.n(); ++n) {
            auto start = data + (n * dims.sizeSingle());
            auto max = std::max_element(start, start + dims.sizeSingle());
            auto pos = std::distance(start, max);
            values[n] = *max;
            positions[n] = pos;
        }
    }
}

template<typename T>
void SingleExtremaFinder<T>::sFindLowest(const CPU &cpu,
    const Dimensions &dims,
    const T *__restrict__ data,
    float *__restrict__ positions,
    T *__restrict__ values) {
    sFindUniversalChecks(dims, data, positions, values);

    if (dims.isPadded()) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
    } else {
        // locate minima
        for (size_t n = 0; n < dims.n(); ++n) {
            auto start = data + (n * dims.sizeSingle());
            auto max = std::min_element(start, start + dims.sizeSingle());
            auto pos = std::distance(start, max);
            values[n] = *max;
            positions[n] = pos;
        }
    }
}

template<typename T>
void SingleExtremaFinder<T>::sFindMax2DAroundCenter(
        const CPU &cpu,
        const Dimensions &dims,
        const T *__restrict__ data,
        float *__restrict__ positions,
        T *__restrict__ values,
        size_t maxDist) {
    sFindUniversal2DAroundCenter(std::greater<T>(),
            std::numeric_limits<T>::lowest(),
            cpu, dims, data, positions, values, maxDist);
}

template<typename T>
void SingleExtremaFinder<T>::sFindLowest2DAroundCenter(
        const CPU &cpu,
        const Dimensions &dims,
        const T *__restrict__ data,
        float *__restrict__ positions,
        T *__restrict__ values,
        size_t maxDist) {
    sFindUniversal2DAroundCenter(std::less<T>(),
            std::numeric_limits<T>::max(),
            cpu, dims, data, positions, values, maxDist);
}

template<typename T>
template<typename C>
void SingleExtremaFinder<T>::sFindUniversal2DAroundCenter(
        const C &comp,
        T startVal,
        const CPU &cpu,
        const Dimensions &dims,
        const T *data,
        float *positions, // can be nullptr
        T * values, // can be nullptr
        size_t maxDist) {
    // check input
    assert(dims.is2D());
    assert( ! dims.isPadded());
    assert(dims.sizeSingle() > 0);
    assert(dims.n() > 0);
    assert(nullptr != data);
    assert((nullptr != positions) || (nullptr != values));
    assert(0 < maxDist);
    const size_t xHalf = dims.x() / 2;
    const size_t yHalf = dims.y() / 2;
    assert((2 * xHalf) > maxDist);
    assert((2 * yHalf) > maxDist);

    const auto min = std::pair<size_t, size_t>(
        std::max((size_t)0, xHalf - maxDist),
        std::max((size_t)0, yHalf - maxDist)
    );

    const auto max = std::pair<size_t, size_t>(
        std::min(dims.x() - 1, xHalf + maxDist),
        std::min(dims.y() - 1, yHalf + maxDist)
    );

    const size_t maxDistSq = maxDist * maxDist;
    for (size_t n = 0; n < dims.n(); ++n) {
        size_t offsetN = n * dims.xyzPadded();
        T extrema = startVal;
        float pos = -1;
        // iterate through the center
        for (size_t y = min.second; y <= max.second; ++y) {
            size_t offsetY = y * dims.x();
            int logicY = (int)y - yHalf;
            size_t ySq = logicY * logicY;
            for (size_t x = min.first; x <= max.first; ++x) {
                int logicX = (int)x - xHalf;
                // continue if the Euclidean distance is too far
                if ((ySq + (logicX * logicX)) > maxDistSq) continue;
                // get current value and update, if necessary
                T tmp = data[offsetN + offsetY + x];
                if (comp(tmp, extrema)) {
                    extrema = tmp;
                    pos = offsetY + x;
                }
            }
        }
        // store results
        if (nullptr != positions) {
            positions[n] = pos;
        }
        if (nullptr != values) {
            values[n] = extrema;
        }
    }
}

// explicit instantiation
template class SingleExtremaFinder<float>;
template class SingleExtremaFinder<double>;

} /* namespace ExtremaFinder */
