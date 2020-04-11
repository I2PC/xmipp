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

#ifndef LIBRARIES_RECONSTRUCTION_AROTATION_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_AROTATION_ESTIMATOR_H_

#include "data/hw.h"
#include "data/dimensions.h"
#include "core/xmipp_error.h"
#include "align_type.h"
#include <vector>
#include <assert.h>
#include <limits>

namespace Alignment {

class RotationEstimationSetting {
public:
    std::vector<HW*> hw;
    AlignType type;
    Dimensions refDims = Dimensions(0);
    Dimensions otherDims = Dimensions(0);
    size_t batch;
    float maxRotDeg;
    bool fullCircle;
    unsigned firstRing;
    unsigned lastRing;
    bool allowTuningOfNumberOfSamples;
    bool allowDataOverwrite; // input data, such as reference or 'other' images can be overwrite by the algorithm. This can be, however, faster

    inline static float getMaxRotation() {
        return 360.f - std::numeric_limits<float>::min();
    }

    inline static unsigned getDefaultLastRing(const Dimensions &d) {
        return (d.x() - 3) / 2; // so that we have some edge around the biggest ring
    }

    inline static unsigned getDefaultFirstRing(const Dimensions &d) {
        return std::max((size_t)2, d.x() / 20);
    }

    inline unsigned getNoOfRings() const {
        return 1 + lastRing - firstRing;
    }

    void check() const {
        if (0 == hw.size()) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "HW contains zero (0) devices");
        }
        if ( ! refDims.isValid()) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "'Reference' dimensions are invalid (contain 0)");
        }
        if ( ! otherDims.isValid()) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "'Other' dimensions are invalid (contain 0)");
        }
        if ( ! refDims.equalExceptNPadded(otherDims)) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Dimensions of the reference and other signals differ");
        }
        if (AlignType::None == type) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "'None' alignment type is set. This is invalid value");
        }
        if ((AlignType::OneToN == type)
                && (1 != refDims.n())) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "More than one reference specified for alignment type 1:N");
        }
        if ((AlignType::MToN == type)
                && (1 == refDims.n())) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Single reference specified for alignment type M:N");
        }
        if (batch > otherDims.n()) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Batch is bigger than number of signals");
        }
        if (0 == batch) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Batch is zero (0)");
        }
        if (0 == maxRotDeg) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "Max rotation is zero (0)");
        }
        if (0 == lastRing) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "Last ring is zero (0)");
        }
        if (0 == firstRing) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "First ring is zero (0)");
        }
        if (lastRing <= firstRing) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "Last ring is bigger (or equal) than first ring");
        }
        if (lastRing >= refDims.x()) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "Last ring is too big");
        }
        if (firstRing >= refDims.x()) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "First ring is too big");
        }
    }
};

template<typename T>
class ARotationEstimator {
public:
    ARotationEstimator() :
        m_isInit(false),
        m_isRefLoaded(false) {};
    // no reference on purpose, we store a copy anyway
    void init(const RotationEstimationSetting settings, bool reuse);

    void loadReference(const T *ref);

    void compute(T *others);

    inline const std::vector<float> &getRotations2D() const {
        return m_rotations2D;
    }

    virtual ~ARotationEstimator() {};

    HW& getHW() const { // FIXME DS remove once we use the new data-centric approach
        assert(m_isInit);
        return *m_settings.hw.at(0);
    }

    inline const RotationEstimationSetting &getSettings() const {
        return m_settings;
    }

protected:
    virtual void check() = 0;

    virtual void init2D() = 0;
    virtual void load2DReferenceOneToN(const T *ref) = 0;
    virtual void computeRotation2DOneToN(T *others) = 0;
    virtual bool canBeReused2D(const RotationEstimationSetting &s) const = 0;


    inline std::vector<float> &getRotations2D() {
        return m_rotations2D;
    }

    inline constexpr bool isInitialized() const {
        return m_isInit;
    }

    inline constexpr bool isRefLoaded() const {
        return m_isRefLoaded;
    }

private:
    RotationEstimationSetting m_settings;

    // computed shifts
    std::vector<float> m_rotations2D;

    // flags
    bool m_isInit;
    bool m_isRefLoaded;

    bool canBeReused(const RotationEstimationSetting &s) const;
};

} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_AROTATION_ESTIMATOR_H_ */
