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

#ifndef LIBRARIES_RECONSTRUCTION_AGEO_TRANSFORMER_H_
#define LIBRARIES_RECONSTRUCTION_AGEO_TRANSFORMER_H_

#include "data/dimensions.h"
#include "core/xmipp_error.h"
#include <memory>
#include "data/hw.h"
#include <vector>

enum class InterpolationDegree { Linear, Cubic };
enum class InterpolationType {
    OneToN, // we have one original item, output are multiple items derived from the original one
    NToN // we have multiple original items, output are multiple items derived from respective original one
};

template<typename T>
class GeoTransformerSettings {
public:
    virtual ~GeoTransformerSettings() = default;
    std::vector<HW*> hw;
    Dimensions dims = Dimensions(0);
    InterpolationDegree degree;
    InterpolationType type;
    bool doWrap;
    T defaultVal;

    virtual void check() const {
        if (0 == hw.size()) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "HW contains zero (0) devices");
        }
        if ( ! dims.isValid()) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Dimensions are invalid");
        }
    }
};

template<typename SettingsType, typename T>
class AGeoTransformer {
    static_assert(std::is_base_of<GeoTransformerSettings<T>, SettingsType>::value,
            "SettingsType must inherit from GeoTransformerSettings");
public:
    AGeoTransformer() :
        m_isInit(false),
        m_isSrcSet(false) {};
    virtual ~AGeoTransformer() = default;

    void init(const SettingsType &s, bool reuse) {
        s.check();
        bool skipAllocation = reuse && this->isInitialized() && canBeReused(s);
        m_settings = s;
        this->initialize( ! skipAllocation);
        this->check();
        m_isInit = true;
    }

    virtual void setSrc(const T *data) = 0;
    virtual const T *getSrc() const = 0;

    virtual T *getDest() const = 0;
    virtual void copySrcToDest() = 0;

    inline const SettingsType &getSettings() const {
        return m_settings;
    }

protected:
    virtual void check() = 0;
    virtual void initialize(bool allocate) = 0;
    virtual bool canBeReused(const SettingsType &s) const = 0;

    inline constexpr bool isInitialized() const {
        return m_isInit;
    }

    inline constexpr bool isSrcSet() const {
        return m_isSrcSet;
    }

    void setIsSrcSet(bool status) {
        m_isSrcSet = status;
    }

private:
    SettingsType m_settings;
    bool m_isInit;
    bool m_isSrcSet;
};


#endif /* LIBRARIES_RECONSTRUCTION_AGEO_TRANSFORMER_H_ */
