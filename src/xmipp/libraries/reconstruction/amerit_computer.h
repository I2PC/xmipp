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

#ifndef LIBRARIES_RECONSTRUCTION_AMERIT_COMPUTER_H_
#define LIBRARIES_RECONSTRUCTION_AMERIT_COMPUTER_H_

#include "data/dimensions.h"
#include "data/hw.h"
#include "core/xmipp_error.h"
#include <vector>

enum class MeritType {
    OneToN // we have one reference item and many others to compare
};

class MeritSettings {
public:
    std::vector<HW*> hw;
    Dimensions refDims = Dimensions(0);
    Dimensions otherDims = Dimensions(0);
    MeritType type;
    bool normalizeResult;

    void check() const {
        if (0 == hw.size()) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "HW contains zero (0) devices");
        }
        if ( ! refDims.isValid()) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "'Ref' dimensions are invalid (contain 0)");
        }
        if ( ! otherDims.isValid()) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "'Other' dimensions are invalid (contain 0)");
        }
        if ( ! refDims.equalExceptNPadded(otherDims)) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Dimensions of the reference and other signals differ");
        }
        if ((MeritType::OneToN == type)
                && (1 != refDims.n())) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "More than one reference specified for type 1:N");
        }
    }
};

template<typename T>
class AMeritComputer {
public:
    AMeritComputer():
        m_isInit(false),
        m_isRefLoaded(false){};
    virtual ~AMeritComputer() = default;

    // no reference on purpose, we store a copy anyway
    void init(const MeritSettings &s, bool reuse) {
        s.check();
        bool skipAllocation = reuse && m_isInit && canBeReused(s);
        m_settings = s;
        this->initialize( ! skipAllocation);
        this->check();
        m_isInit = true;
    }

    inline const MeritSettings &getSettings() const {
        return m_settings;
    }

    virtual void loadReference(const T *ref) = 0;

    virtual void compute(T *others) = 0;

    inline const std::vector<float> &getFiguresOfMerit() const {
        return m_figures;
    }

protected:
    virtual void check() = 0;
    virtual bool canBeReused(const MeritSettings &s) const = 0;
    virtual void initialize(bool allocate) = 0;

    inline constexpr bool isInitialized() const {
        return m_isInit;
    }

    inline constexpr bool isRefLoaded() const {
        return m_isRefLoaded;
    }

    void setIsRefLoaded(bool status) {
        m_isRefLoaded = status;
    }

    inline std::vector<float> &getFiguresOfMerit() {
        return m_figures;
    }

private:
    MeritSettings m_settings;
    std::vector<float> m_figures;

    // flags
    bool m_isInit;
    bool m_isRefLoaded;
};

#endif /* LIBRARIES_RECONSTRUCTION_AMERIT_COMPUTER_H_ */
