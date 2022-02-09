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

#include "correlation_computer.h"

template<typename T>
template<bool NORMALIZE>
void CorrelationComputer<T>::computeOneToN(T *others) {
    const auto &s = this->getSettings();
    const size_t n = s.otherDims.n();
    const size_t z = s.otherDims.z();
    const size_t y = s.otherDims.y();
    const size_t x = s.otherDims.x();

    auto &res = this->getFiguresOfMerit();
    res.resize(n); // allocate all positions, so that we can access it our of order
    auto workload = [&](int id, size_t signalId){
        auto ref = MultidimArray<T>(1, z, y, x, const_cast<T*>(m_ref)); // removing const, but data should not be changed
        T * address = others + signalId * s.otherDims.sizeSingle();
        auto other = MultidimArray<T>(1, z, y, x, address);
        if (NORMALIZE) {
            res.at(signalId) = correlationIndex(ref, other);
        } else {
            res.at(signalId) = fastCorrelation(ref, other);
        }
    };

    auto futures = std::vector<std::future<void>>();
    for (size_t i = 0; i < n; ++i) {
        futures.emplace_back(m_threadPool.push(workload, i));
    }
    for (auto &f : futures) {
        f.get();
    }
}

template<typename T>
void CorrelationComputer<T>::compute(T *others) {
    bool isReady = this->isInitialized() && this->isRefLoaded();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() and load reference");
    }
    const auto &s = this->getSettings();
    switch(s.type) {
        case MeritType::OneToN: {
            if (s.normalizeResult) {
                computeOneToN<true>(others);
            } else {
                computeOneToN<false>(others);
            }
            break;
        }
        default:
            REPORT_ERROR(ERR_NOT_IMPLEMENTED, "This case is not implemented");
    }
}

template<typename T>
void CorrelationComputer<T>::loadReference(const T *ref) {
    m_ref = ref; // just copy pointer
    this->setIsRefLoaded(nullptr != ref);
}

template<typename T>
void CorrelationComputer<T>::initialize(bool doAllocation) {
    const auto &s = this->getSettings();
    release();

    for (auto &hw : s.hw) {
        if ( ! dynamic_cast<CPU*>(hw)) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Instance of CPU is expected");
        }
    }
//    m_threadPool.resize(s.hw.size()); // FIXME DS set to requested number of thread
    m_threadPool.resize(CPU::findCores());
}

template<typename T>
void CorrelationComputer<T>::release() {
    setDefault();
}

template<typename T>
void CorrelationComputer<T>::setDefault() {
    m_ref = nullptr;
    m_threadPool.resize(0);
}

template<typename T>
bool CorrelationComputer<T>::canBeReused(const MeritSettings &s) const {
    bool result = true;
    if ( ! this->isInitialized()) {
        return false;
    }
    auto &sOrig = this->getSettings();
    result = result && sOrig.type == s.type;
    result = result && (sOrig.otherDims.size() >= s.otherDims.size()); // previously, we needed more space

    return result;
}

template<typename T>
void CorrelationComputer<T>::check() {
    // so far nothing to do
}

// explicit instantiation
template class CorrelationComputer<float>;
template class CorrelationComputer<double>;
