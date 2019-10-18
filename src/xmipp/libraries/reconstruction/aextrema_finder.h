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

#ifndef LIBRARIES_RECONSTRUCTION_AEXTREMA_FINDER_H_
#define LIBRARIES_RECONSTRUCTION_AEXTREMA_FINDER_H_

#include "data/dimensions.h"
#include "data/hw.h"
#include "data/point3D.h"
#include "core/xmipp_error.h"
#include <vector>
#include <cassert>
#include <limits>

namespace ExtremaFinder {
// FIXME DS we should have search type Min, Max, MaxAbs, Lowest, Custom
// FIXME DS we should have search location Entire, NearCenter, AroundCenter, Window, AroundWindow
enum class SearchType {
    Lowest, // in the whole signal (for each signal)
    Max, // in the whole signal (for each signal)
    MaxAroundCenter, // for each signal, search a circular area around center
    MaxNearCenter, // for each signal, search a square area around center
    LowestAroundCenter // for each signal, search a circular area around center
};

enum class ResultType {
    Value,
    Position,
    Both
};

class ExtremaFinderSettings {
public:
    std::vector<HW*> hw;
    SearchType searchType;
    ResultType resultType;
    Dimensions dims = Dimensions(0);
    size_t batch = 0;
    float maxDistFromCenter = 0.f;

    Point3D<size_t> getCenter() const {
        return Point3D<size_t>(dims.x() / 2, dims.y() / 2, dims.z() / 2);
    }

    void check() const {
        if (0 == hw.size()) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "HW contains zero (0) devices");
        }
        if ( ! dims.isValid()) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Dimensions are invalid (contain 0)");
        }
        if (0 == batch) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Batch is zero (0)");
        }
        if ((SearchType::MaxAroundCenter == searchType)
                || (SearchType::LowestAroundCenter == searchType)) {
            const auto center = getCenter();
            if (0 == maxDistFromCenter) {
                REPORT_ERROR(ERR_LOGIC_ERROR, "'maxDistFromCenter' is set to zero (0)");
            }
            if (dims.is1D()) {
                if (maxDistFromCenter >= center.x) {
                    REPORT_ERROR(ERR_LOGIC_ERROR, "'maxDistFromCenter' is bigger than half of the signal's X dimension");
                }
            } else if (dims.is2D()) {
                if ((maxDistFromCenter >= center.x)
                        || (maxDistFromCenter >= center.y)) {
                    REPORT_ERROR(ERR_LOGIC_ERROR, "'maxDistFromCenter' is bigger than half of the signal's X or Y dimensions");
                }
            } else {
                if ((maxDistFromCenter >= center.x)
                        || (maxDistFromCenter >= center.y)
                        || (maxDistFromCenter >= center.z)) {
                    REPORT_ERROR(ERR_LOGIC_ERROR, "'maxDistFromCenter' is bigger than half of the signal's X, Y or Z dimensions");
                }
            }
        }
    }
};

template<typename T>
class AExtremaFinder {
public:
    AExtremaFinder() :
        m_isInit(false) {};

    virtual ~AExtremaFinder() {};

    void init(const ExtremaFinderSettings &settings, bool reuse);

    void find(const T *data);

    HW& getHW() const { // FIXME DS remove once we use the new data-centric approach
        assert(m_isInit);
        return *m_settings.hw.at(0);
    }

    inline const ExtremaFinderSettings &getSettings() const {
        return m_settings;
    }

    inline const std::vector<T> &getValues() const {
        return m_values;
    }

    inline const std::vector<float> &getPositions() const {
        return m_positions;
    }

protected:
    virtual void check() const = 0;

    virtual void initMax() = 0;
    virtual void findMax(const T *data) = 0;
    virtual bool canBeReusedMax(const ExtremaFinderSettings &s) const = 0;

    virtual void initLowest() = 0;
    virtual void findLowest(const T *data) = 0;
    virtual bool canBeReusedLowest(const ExtremaFinderSettings &s) const = 0;

    virtual void initMaxAroundCenter() = 0;
    virtual void findMaxAroundCenter(const T *data) = 0;
    virtual bool canBeReusedMaxAroundCenter(const ExtremaFinderSettings &s) const = 0;

    virtual void initLowestAroundCenter() = 0;
    virtual void findLowestAroundCenter(const T *data) = 0;
    virtual bool canBeReusedLowestAroundCenter(const ExtremaFinderSettings &s) const = 0;

    inline std::vector<T> &getValues() {
        return m_values;
    }

    inline std::vector<float> &getPositions() {
        return m_positions;
    }

    inline constexpr bool isInitialized() const {
        return m_isInit;
    }

private:
    ExtremaFinderSettings m_settings;

    // results
    std::vector<T> m_values;
    std::vector<float> m_positions; // absolute, 0 based indices

    // flags
    bool m_isInit;

    bool canBeReused(const ExtremaFinderSettings &settings) const;
};

} /* namespace ExtremaFinder */

#endif /* LIBRARIES_RECONSTRUCTION_AEXTREMA_FINDER_H_ */
