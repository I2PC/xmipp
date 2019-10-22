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

#include "aextrema_finder.h"

namespace ExtremaFinder {

template<typename T>
void AExtremaFinder<T>::init(const ExtremaFinderSettings &settings, bool reuse) {
    // check that settings is not completely wrong
    settings.check();
    bool skipInit = m_isInit && reuse && this->canBeReused(settings);
    // set it
    m_settings = ExtremaFinderSettings(settings);
    if ( ! skipInit) {
        // initialize
        switch (m_settings.searchType) {
            case SearchType::Max:
                this->initMax();
                break;
            case SearchType::Lowest:
                this->initLowest();
                break;
            case SearchType::MaxAroundCenter:
                this->initMaxAroundCenter();
                break;
            case SearchType::LowestAroundCenter:
                this->initLowestAroundCenter();
                break;
            default: REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
        }
        // check that there's no logical problem
        this->check();
        // no issue found, we're good to go
        m_isInit = true;
    }
}

template<typename T>
bool AExtremaFinder<T>::canBeReused(const ExtremaFinderSettings &s) const {
    switch (s.searchType) {
        case SearchType::Max: return this->canBeReusedMax(s);
        case SearchType::Lowest: return this->canBeReusedLowest(s);
        case SearchType::MaxAroundCenter: return this->canBeReusedMaxAroundCenter(s);
        case SearchType::LowestAroundCenter: return this->canBeReusedLowestAroundCenter(s);
        default: REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
    }
}

template<typename T>
void AExtremaFinder<T>::find(const T *data) {
    if ((ResultType::Position == m_settings.resultType)
        || (ResultType::Both == m_settings.resultType)) {
        m_positions.resize(m_settings.dims.n(), -1.f);
    }
    if ((ResultType::Value == m_settings.resultType)
        || (ResultType::Both == m_settings.resultType)) {
        m_values.resize(m_settings.dims.n(), std::numeric_limits<T>::lowest());
    }
    switch (m_settings.searchType) {
        case SearchType::Max: return this->findMax(data);
        case SearchType::Lowest: return this->findLowest(data);
        case SearchType::MaxAroundCenter: return this->findMaxAroundCenter(data);
        case SearchType::LowestAroundCenter: return this->findLowestAroundCenter(data);
        default: REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
    }
}

// explicit instantiation
template class AExtremaFinder<float>;
template class AExtremaFinder<double>;

} /* namespace ExtremaFinder */
