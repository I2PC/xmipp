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
enum class InterpolationType { OneToN, NToN };

class AInterpolationMethod {
public:
    virtual ~AInterpolationMethod() = default;
};

template<typename T>
class BSplineInterpolation : public AInterpolationMethod {
public:
    T defaultVal;
};

class GeoTransformerSetting {
public:
    std::vector<HW*> hw;
    Dimensions dims = Dimensions(0);
    InterpolationDegree degree;
    AInterpolationMethod *method;
    bool createReferenceCopy;
    InterpolationType type;
    bool doWrap;

    void check() const {
        if (0 == hw.size()) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "HW contains zero (0) devices");
        }
        if ( ! dims.isValid()) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Dimensions are invalid");
        }
        if (nullptr == method) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Interpolation type is not set");
        }
    }
};

template<typename T>
class AGeoTransformer {
public:
    AGeoTransformer() :
        m_isInit(false),
        m_isOrigLoaded(false) {};
    virtual ~AGeoTransformer() = default;

    void init(const GeoTransformerSetting &s, bool reuse);

    virtual void setOriginal(const T *data) = 0;
    virtual const T *getOriginal() const = 0;

    virtual T *getCopy() const = 0;
    virtual void copyOriginalToCopy() = 0;

    virtual T *interpolate(const std::vector<float> &matrices) = 0; // each 3x3 values are a single matrix

    inline const GeoTransformerSetting &getSettings() const {
        return m_settings;
    }

protected:
    virtual void check() = 0;
    virtual void init(bool allocate) = 0;
    virtual bool canBeReused(const GeoTransformerSetting &s) const = 0;

    inline constexpr bool isInitialized() const {
        return m_isInit;
    }

    inline constexpr bool isOrigLoaded() const {
        return m_isOrigLoaded;
    }

    void setIsOrigLoaded(bool status) {
        m_isOrigLoaded = status;
    }

private:
    GeoTransformerSetting m_settings;
    bool m_isInit;
    bool m_isOrigLoaded;
};


#endif /* LIBRARIES_RECONSTRUCTION_AGEO_TRANSFORMER_H_ */
