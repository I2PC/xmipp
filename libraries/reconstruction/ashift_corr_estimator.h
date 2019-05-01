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

#ifndef LIBRARIES_RECONSTRUCTION_ASHIFT_CORR_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_ASHIFT_CORR_ESTIMATOR_H_

#include "ashift_estimator.h"
#include "data/fft_settings_new.h"
#include "core/xmipp_error.h"
#include "data/hw.h"
namespace Alignment {

template<typename T>
class AShiftCorrEstimator : public AShiftEstimator<T> {
public:
    AShiftCorrEstimator() {
        setDefault();
    }
    virtual ~AShiftCorrEstimator() {
        release();
    }

    virtual void init2D(const HW &hw, AlignType type,
            const FFTSettingsNew<T> &dims, size_t maxShift,
            bool includingBatchFT, bool includingSingleFT) = 0;

    virtual void load2DReferenceOneToN(const std::complex<T> *h_ref) = 0;

    virtual void computeCorrelations2DOneToN(
            std::complex<T> *inOut, bool center) = 0;

    virtual void computeCorrelations2DOneToN(
            const HW &hw,
            std::complex<T> *inOut,
            const std::complex<T> *ref,
            size_t xDim, size_t yDim, size_t nDim,
            bool center=false) = 0;

    static std::vector<T> findMaxAroundCenter(
            const T *data,
            const Dimensions &dims,
            const Point2D<size_t> &maxShift,
            std::vector<Point2D<int>> &shifts);

    static std::vector<T> findMaxAroundCenter(
            const T *data,
            const Dimensions &dims,
            size_t maxShift,
            std::vector<Point2D<int>> &shifts);
    virtual void release() override;
protected:
    const FFTSettingsNew<T> *m_settingsInv; // FIXME DS rename
    size_t m_maxShift;
    size_t m_centerSize;
    AlignType m_type;

    // helper objects / memory
    T *m_h_centers;

    // flags
    bool m_includingBatchFT;
    bool m_includingSingleFT;
    bool m_is_single_FD_loaded;
    bool m_isInit;

    virtual void setDefault();
    void init2D(AlignType type,
            const FFTSettingsNew<T> &dims, size_t maxShift,
            bool includingBatchFT, bool includingSingleFT);

    virtual void check();
    virtual void init2DOneToN();
};

} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_ASHIFT_CORR_ESTIMATOR_H_ */
