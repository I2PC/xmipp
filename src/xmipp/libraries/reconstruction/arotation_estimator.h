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

namespace Alignment {

template<typename T>
class ARotationEstimator {
public:
    ARotationEstimator() {
        setDefault();
    }

    void init(const HW &hw, AlignType type,
       const Dimensions &dims, size_t batch, float maxRotDeg);

    void loadReference(const T *ref);

    void compute(T *others);

    constexpr bool isInitialized() const {
        return m_isInit;
    }

    constexpr AlignType getAlignType() const {
        return m_type;
    }

    constexpr Dimensions getDimensions() const {
        return *m_dims;
    }

    inline std::vector<float> getRotations2D() {
        if ( ! m_is_rotation_computed) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Rotation has not been yet computed");
        }
        return m_rotations2D;
    }

    virtual void release();
    virtual ~ARotationEstimator() {
        release();
    }

protected:
    // various
    AlignType m_type;
    const Dimensions *m_dims;
    size_t m_batch;
    float m_maxRotationDeg;

    // computed shifts
    std::vector<float> m_rotations2D;

    // flags
    bool m_is_ref_loaded;
    bool m_is_rotation_computed;
    bool m_isInit;

    virtual void setDefault();
    virtual void check();

    virtual void init2D(const HW &hw) = 0;
    virtual void load2DReferenceOneToN(const T *ref) = 0;
    virtual void computeRotation2DOneToN(T *others) = 0;
};

} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_AROTATION_ESTIMATOR_H_ */
