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

#ifndef LIBRARIES_RECONSTRUCTION_ASHIFT_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_ASHIFT_ESTIMATOR_H_

#include "data/hw.h"
#include "data/dimensions.h"
#include "data/point2D.h"
#include "core/xmipp_error.h"
#include "align_type.h"
#include <vector>

namespace Alignment {

template<typename T>
class AShiftEstimator {
public:
    AShiftEstimator() {
        setDefault();
    }
    virtual ~AShiftEstimator() {
        release();
    }

    virtual void init2D(const std::vector<HW*> &hw, AlignType type,
               const Dimensions &dims, size_t batch, size_t maxShift) = 0;

    virtual void load2DReferenceOneToN(const T *ref) = 0;

    virtual void computeShift2DOneToN(T *others) = 0; // FIXME DS it should erase m_shifts2D

    inline std::vector<Point2D<float>> getShifts2D() {
        if ( ! m_is_shift_computed) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Shift has not been yet computed or it has been already retrieved");
        }
        auto cpy = std::vector<Point2D<float>>();
        cpy.swap(m_shifts2D);
        m_is_shift_computed = false;
        return cpy;
    }

    virtual void release();

    constexpr bool isInitialized() const {
        return m_isInit;
    }

    constexpr Dimensions getDimensions() const {
        return *m_dims;
    }

    constexpr AlignType getAlignType() const {
        return m_type;
    }

    virtual HW& getHW() const = 0;

protected:
    // various
    AlignType m_type;
    const Dimensions *m_dims;
    size_t m_batch;
    size_t m_maxShift;

    // computed shifts
    std::vector<Point2D<float>> m_shifts2D;

    // flags
    bool m_is_ref_loaded;
    bool m_is_shift_computed;
    bool m_isInit;

    virtual void setDefault();
    virtual void init2D(AlignType type, const Dimensions &dims,
               size_t batch, size_t maxShift);
    virtual void check();
};

} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_ASHIFT_ESTIMATOR_H_ */
